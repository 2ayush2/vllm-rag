from fastapi import APIRouter
from api.dependencies import llm, rag

router = APIRouter()

@router.get("/status")
async def get_status():
    """Returns the current status of the LLM and RAG services."""
    return {
        "status": "online",
        "model_name": llm.model_name,
        "rag": {
            "active": rag.db is not None,
            "current_document": rag.current_doc,
            "embedding_model": rag.embeddings.model_name if hasattr(rag.embeddings, 'model_name') else "unknown"
        }
    }
