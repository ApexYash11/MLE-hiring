"""Central configuration for the support triage pipeline."""

MODEL = "deepseek-chat"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
TEMPERATURE = 0
SEED = 42
MAX_TOKENS = 1024

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 5
RETRIEVAL_CANDIDATE_K = 20
SEMANTIC_INJECTION_THRESHOLD = 0.72
MAX_CONTEXT_TOKENS = 5500
MAX_RETRIEVAL_TOKENS = 3500
MAX_CONVERSATION_TOKENS = 1500
PRODUCT_BOOST_FACTOR = 1.3

SAFE_ESCALATION_CONFIDENCE = 0.92
CONTRADICTION_MAX_CONFIDENCE = 0.60
MAX_LLM_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2

FALLBACK_MODEL_CHAIN = [
    "openrouter/free",
    "deepseek-chat",
]

PER_MODEL_TIMEOUT = 120
