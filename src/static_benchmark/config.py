"""Local configuration for benchmark scripts.

Fill in your credentials here if you want to use LLM-as-judge with OpenAI.
This file is intentionally simple so you can edit it directly.
"""

# OpenAI credentials for judge mode.
OPENAI_API_KEY = ""
OPENAI_BASE_URL = None

# Judge model to use with the OpenAI Responses API.
# A current balance pick is "gpt-5.6"; you can also pin a specific variant.
OPENAI_JUDGE_MODEL = "gpt-5.6"

# Embedding model for cosine similarity.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Generation defaults for local Hugging Face runs.
DEFAULT_GENERATION_MODEL = "google/gemma-3-4b-it"
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE = 0.2
