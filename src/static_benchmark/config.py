"""Local configuration for benchmark scripts.

Fill in your credentials here if you want to use LLM-as-judge with OpenAI.
This file is intentionally simple so you can edit it directly.
"""

# OpenAI credentials for judge mode.
# For plain OpenAI:
#   OPENAI_API_KEY = "sk-..."
#   OPENAI_BASE_URL = None
#
# For Azure OpenAI:
#   OPENAI_USE_AZURE = True
#   OPENAI_API_KEY = "..."
#   OPENAI_AZURE_ENDPOINT = "https://your-resource.openai.azure.com"
#   OPENAI_API_VERSION = "2025-01-01-preview"
#   OPENAI_JUDGE_MODEL = "your-deployment-name"
OPENAI_API_KEY = ""
OPENAI_BASE_URL = None
OPENAI_USE_AZURE = False
OPENAI_AZURE_ENDPOINT = None
OPENAI_API_VERSION = None

# Judge model to use with the OpenAI Responses API.
# For Azure OpenAI, this should be your deployment name.
OPENAI_JUDGE_MODEL = "gpt-5.6"

# Embedding model for cosine similarity.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Generation defaults for local Hugging Face runs.
DEFAULT_GENERATION_MODEL = "google/gemma-3-4b-it"
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE = 0.2
