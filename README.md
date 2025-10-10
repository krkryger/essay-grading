# Automated essay grading pipeline

Desc. here

### Installation
The pipeline needs Python (3.13 recommended). To install dependencies, use Conda or pip.

With pip: `pip install -r requirements.txt`

### Usage
All of the functions are tucked in `src/grading.py`. You can import them into your own script or noteboox, like in `usage-example.ipynb`.

Currently, the pipeline supports calling OpenAI, Google and Anthropic APIs. To use them, you need to provide API keys in a file called `.env` (see `.env.example` for how it should look like).

The pipeline expects an input file and a grading template in the `./data/9kl` folder. To configure LLM instructions, use the config file in the same folder.
