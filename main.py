from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.health_api import router as health_router
from api.response_api import router as response_router
# from api.rag_api import router as rag_router

app = FastAPI(
    title="test flowcess",
    description="Direct LLM interface and RAG endpoints",
    version="1.2.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(response_router)
# app.include_router(rag_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
