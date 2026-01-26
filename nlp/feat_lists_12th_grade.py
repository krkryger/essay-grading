# 12th grade feature lists with omissions due to multicollinearity

# Punctuation
punct_feats = ["Asp_42_Average", "unnecessary_punct", "missing_punct", "replaced_punct",
    "replaced_punct_ratio"]

# Orthography and morphology
orth_feats = ["Asp_43_Average", "spell_error_word_ratio", "replaced_words", "replaced_word_ratio", 
    "whitespace_errors", "misc_errors", "misc_error_ratio", "errors_per_word"]

# Structuring and formatting
struct_feats = ["Asp_34_Average", "paragraph_count", "mean_word_count", "diff_word_count",
    "mean_sent_count", "std_sent_count", "replaced_words", "replaced_word_ratio", 
    "spell_error_word_ratio"]

# Syntax
sent_feats = ["Asp_40_Average", "wordorder_errors", "missing_words", "unnecessary_words", 
    "misc_errors", "misc_error_ratio", "word_len", "sent_len", "LIX", "LD", "S_abstr", 
	"rare_3000", "A", "D", "I", "J", "K", "P", "V", "J_Sub", "K_Post", "Nom", "Gen", "Par", 
    "AddIll", "Ill", "Ine", "Ela", "All", "Ade", "Abl", "Tra", "Ter", "Ess", "Abe", "Com",
    "Plur", "S_cases", "A_cases", "A_Plur", "A_Cmp", "A_Sup", "P_cases", "P_Plur", "P_IntRel", 
    "V_Fin", "V_Ind", "V_Cnd", "V_Imp", "V_Prs1", "V_Prs2", "V_Prs3", "V_Pres", "V_Past", 
    "V_Sing", "V_Plur", "V_Neg", "V_Pass", "V_NonFin", "V_Inf", "V_Part", "V_Conv"]

# Vocabulary
word_feats = ["Asp_28_Average", "replaced_words", "replaced_word_ratio", "missing_words", 
    "unnecessary_words", "wordorder_errors", "wordorder_error_ratio", "misc_errors", 
	"misc_error_ratio", "lemma_count", "Maas", "MTLD", "CVV", "A_TTR", "D_TTR", "J_TTR", 
	"K_TTR", "P_TTR", "S_TTR", "V_TTR", "LD", "S_abstr", "rare_3000", "A", "D", "I", "K", 
	"P", "V", "P_Prs", "P_Reflex", "P_Dem", "P_Ind", "P_IntRel", "V_Prs1", "V_Prs2", "V_Prs3", 
	"word_len"]