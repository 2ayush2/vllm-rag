from core.model import LocalLLM
from services.rag import NepaliRAG

# Singletons for the models and services
llm = LocalLLM()
rag = NepaliRAG()
