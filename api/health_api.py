from fastapi import APIRouter
from api.dependencies import llm

router = APIRouter()

@router.get("/status")
async def get_status():
    return {
        "status": "online",
        "model_name": llm.model_name,
        "rag_context_active": False  # Disabled temporarily
    }
