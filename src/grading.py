"""
Minimal, reusable helpers to:
- load rubric + essays
- build essay-aspect prompts
- call OpenAI / Gemini / Anthropic with adaptive backoff (increasing max_tokens on retries)

Usage from a notebook or CLI:
    from grading import (
        load_rubric, load_essays, build_essay_aspect_prompts,
        make_grader_with_adaptive_tokens, grade_one_row
    )

    mudel = load_rubric("data/9kl/hindamismudel_9kl.json")
    essays = load_essays("data/9kl/Kirjandid_9.klass_(9887).xlsx")
    essay_aspect_prompts = build_essay_aspect_prompts(essays, mudel)

    row = essay_aspect_prompts.iloc[0]
    grade_single_prompt = make_grader_with_adaptive_tokens(initial_max_tokens=512)
    out = grade_single_prompt(provider="google", model="gemini-2.5-pro", prompt=row["prompt"])
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import json
import backoff
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env if present (safe no-op if absent)
load_dotenv()

__all__ = [
    # Data loading / building
    "load_rubric",
    "load_essays",
    "build_essay_aspect_prompts",
    # API caller + helpers
    "make_grader_with_adaptive_tokens",
    "grade_one_row",
    "grade_one_aspect",
    "grade_all",
]


# =========================
# Data loading & preparation
# =========================

# add near other imports/utilities
def load_prompt_config(path: str | Path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def load_rubric(path: str | Path) -> Dict[str, str]:
    """Load rubric JSON (mudel)."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_essay_text(text: Any) -> Any:
    """Normalize spacing and newlines similar to the original R cleaning."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"[ ]+", " ", text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(\n[ ]*){3,}", "\n\n", text)
    text = re.sub(r"\n+$", "", text)
    return text


def load_essays(path: str | Path) -> pd.DataFrame:
    """Load essays Excel and clean the 'Kirjand' column."""
    df = pd.read_excel(path)
    if "Sooritaja_id" not in df.columns:
        raise KeyError("Expected column 'Sooritaja_id' not found in the Excel data.")
    if "Kirjand" in df.columns:
        df["Kirjand"] = df["Kirjand"].apply(_clean_essay_text)
    return df


def build_essay_aspect_prompts(essays: pd.DataFrame, mudel: dict, config: dict) -> pd.DataFrame:
    """
    Build cross-join of essays x aspects using external prompt config.
    Expects config keys:
      - student_task_instructions
      - grading_scale_instructions
      - grader_task_template
      - response_format_instructions
      - aspects[]  (each with: asp_id, label, rubric_key)
    """
    essays_min = essays[["Sooritaja_id", "Kirjand"]].copy()

    task_intro = config["grader_task_template"].format(
        student_task_instructions=config["student_task_instructions"]
    )
    grading_scale = config["grading_scale_instructions"]
    response_fmt = config["response_format_instructions"]


    # Make aspect table and pull rubric text from mudel via rubric_key
    aspects = pd.DataFrame(config["aspects"])
    aspects = aspects.rename(columns={"asp_id": "Asp", "label": "tunnus"})
    aspects["mudel"] = aspects["rubric_key"].map(mudel).fillna("")

    # Base prompt per aspect
    aspects["prompt_base"] = (
        task_intro + " " + aspects["tunnus"] + ". " + grading_scale
        + "\n\n" + aspects["mudel"] + "\n\n" + response_fmt + "\n"
    )

    # Cross-join essays × aspects
    essays_min["_join"] = 1
    aspects["_join"] = 1
    df = essays_min.merge(aspects, on="_join").drop(columns="_join")

    # Final prompt
    df["prompt"] = (
        df["prompt_base"] + df["Kirjand"].fillna("")
        + '\n"""\n\nNow provide Seletus and Hinne for ' + df["tunnus"].fillna("") + "!"
    )

    return (
        df[["Sooritaja_id", "Asp", "tunnus", "prompt"]]
        .sort_values(["Sooritaja_id", "Asp"])
        .reset_index(drop=True)
    )


# =========================
# Universal API caller (LLMs)
# =========================

# Lazy clients
_OPENAI = None
_GOOGLE = None
_ANTHROPIC = None


def _openai_client():
    global _OPENAI
    if _OPENAI is None:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _OPENAI = OpenAI(api_key=key)
    return _OPENAI


def _google_client():
    global _GOOGLE
    if _GOOGLE is None:
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set")
        genai.configure(api_key=key)
        _GOOGLE = genai
    return _GOOGLE


def _anthropic_client():
    global _ANTHROPIC
    if _ANTHROPIC is None:
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _ANTHROPIC = anthropic.Anthropic(api_key=key)
    return _ANTHROPIC


def _parse_score(text: str):
    """Extract last integer/decimal from text, if any."""
    if not isinstance(text, str):
        return None
    m = re.findall(r"\d+(?:\.\d+)?", text)
    return float(m[-1]) if m else None


def _gemini_extract_text(resp) -> str:
    """Safely collect text from Gemini response without touching resp.text (can raise on MAX_TOKENS)."""
    parts = []
    for c in getattr(resp, "candidates", []) or []:
        content = getattr(c, "content", None)
        if not content:
            continue
        for p in getattr(content, "parts", []) or []:
            t = getattr(p, "text", None)
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return "\n".join(parts).strip()


def _single_attempt_call(
    *, provider: str, model: str, prompt: str, max_tokens: int, temperature: float = 0.0
) -> Dict[str, Any]:
    """Single attempt to call an LLM provider. No backoff here."""
    provider = provider.lower()
    system_prompt = "You are a careful, consistent grading assistant."

    if provider == "openai":
        client = _openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()

    elif provider == "google":
        genai = _google_client()
        gmodel = genai.GenerativeModel(model)
        resp = gmodel.generate_content(
            system_prompt + "\n\n" + prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "candidate_count": 1,
            },
            safety_settings=None,
        )
        text = _gemini_extract_text(resp)
        if not text:
            # Try to surface finish_reason for debugging
            fr = None
            try:
                fr = getattr(getattr(resp, "candidates", [None])[0], "finish_reason", None)
            except Exception:
                pass
            raise ValueError(f"Gemini returned no text (finish_reason={fr}).")

    elif provider == "anthropic":
        client = _anthropic_client()
        resp = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        chunks = []
        for part in getattr(resp, "content", []) or []:
            if getattr(part, "text", None):
                chunks.append(part.text)
        text = ("\n".join(chunks)).strip()

    else:
        raise ValueError("provider must be 'openai', 'google', or 'anthropic'.")

    return {"raw_text": text, "parsed_score": _parse_score(text)}


def make_grader_with_adaptive_tokens(
    initial_max_tokens: int = 512, growth_factor: float = 2.0, max_tokens_cap: int = 4096
):
    """
    Returns a function grade_single_prompt(...) that:
      - retries with backoff
      - multiplies max_tokens on each retry (capped)
    """
    state = {"max_tokens": int(initial_max_tokens)}

    def _on_backoff(_details):
        state["max_tokens"] = min(int(state["max_tokens"] * growth_factor), int(max_tokens_cap))

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_time=60,
        jitter=backoff.full_jitter,
        on_backoff=_on_backoff,
    )
    def grade_single_prompt(*, provider: str, model: str, prompt: str, temperature: float = 0.0):
        return _single_attempt_call(
            provider=provider,
            model=model,
            prompt=prompt,
            max_tokens=state["max_tokens"],
            temperature=temperature,
        )

    return grade_single_prompt


# =========================
# Tiny DataFrame helpers
# =========================

def grade_one_row(row: pd.Series, *, provider: str, model: str, initial_max_tokens: int = 512) -> Dict[str, Any]:
    """
    Grade a single row from essay_aspect_prompts (expects columns: Sooritaja_id, Asp, tunnus, prompt).
    Returns dict with raw_text, parsed_score, provider, model + ids.
    """
    grade_single_prompt = make_grader_with_adaptive_tokens(initial_max_tokens=initial_max_tokens)
    out = grade_single_prompt(provider=provider, model=model, prompt=row["prompt"])
    return {
        "Sooritaja_id": row["Sooritaja_id"],
        "Asp": row["Asp"],
        "tunnus": row["tunnus"],
        "raw_text": out["raw_text"],
        "parsed_score": out["parsed_score"],
        "provider": provider,
        "model": model,
    }


def grade_one_aspect(
    essay_aspect_prompts: pd.DataFrame,
    aspect_id: str,
    *,
    provider: str,
    model: str,
    limit: Optional[int] = None,
    initial_max_tokens: int = 512,
) -> pd.DataFrame:
    """
    Grade all rows for a given aspect_id (e.g., "Asp_35").
    Optionally limit the number of rows to keep costs safe.
    """
    subset = essay_aspect_prompts.query("Asp == @aspect_id")
    if limit is not None:
        subset = subset.head(int(limit))

    grade_single_prompt = make_grader_with_adaptive_tokens(initial_max_tokens=initial_max_tokens)

    results = []
    for _, r in tqdm(subset.iterrows(), total=(limit or len(essay_aspect_prompts))):
        out = grade_single_prompt(provider=provider, model=model, prompt=r["prompt"])
        results.append({
            "Sooritaja_id": r["Sooritaja_id"],
            "Asp": r["Asp"],
            "tunnus": r["tunnus"],
            "raw_text": out["raw_text"],
            "parsed_score": out["parsed_score"],
            "provider": provider,
            "model": model,
        })
    return pd.DataFrame(results)


def grade_all(
    essay_aspect_prompts: pd.DataFrame,
    *,
    provider: str,
    model: str,
    limit: Optional[int] = None,
    initial_max_tokens: int = 512,
) -> pd.DataFrame:
    """
    Grade across all aspects x all essays. Use `limit` to control spend.
    """
    df = essay_aspect_prompts if limit is None else essay_aspect_prompts.head(int(limit))
    grade_single_prompt = make_grader_with_adaptive_tokens(initial_max_tokens=initial_max_tokens)

    out_rows = []
    for _, r in tqdm(df.iterrows(), total=(limit or len(essay_aspect_prompts))):
        out = grade_single_prompt(provider=provider, model=model, prompt=r["prompt"])
        out_rows.append({
            "Sooritaja_id": r["Sooritaja_id"],
            "Asp": r["Asp"],
            "tunnus": r["tunnus"],
            "raw_text": out["raw_text"],
            "parsed_score": out["parsed_score"],
            "provider": provider,
            "model": model,
        })
    return pd.DataFrame(out_rows)