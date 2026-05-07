import os

# vLLM Configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "./models/qwen3-4b-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")

# RAG Configuration
DB_DIR = os.getenv("DB_DIR", "./database/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Logging Configuration
LOG_DIR = os.getenv("LOG_DIR", "logs")
API_LOG_FILE = os.path.join(LOG_DIR, "api.log")

# Create log directory if it doesn't exist
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
