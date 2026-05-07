import json, uuid, time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from api.dependencies import rag
from schemas.chat import ChatRequest
from utils.token_counter import count_tokens
from utils.logging_utils import setup_logger

logger = setup_logger("rag_api_logger", "logs/api.log")
router = APIRouter()

@router.post("/v1/rag/completions")
async def handle_rag_request(req_body: ChatRequest, request: Request):
    req_id, start_time = str(uuid.uuid4()), time.time()
    try:
        messages = req_body.get_messages()
        user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if not user_msg: raise HTTPException(status_code=400, detail="No user message found")
        logger.info(f"[{req_id}] Incoming RAG Request: {req_body.model_dump_json()}")
        in_tokens = count_tokens(user_msg + sys_msg)

        if req_body.stream:
            def generate_rag_stream():
                full_text = ""
                for chunk in rag.query(user_msg, sys_msg):
                    if not chunk: continue
                    full_text += chunk
                    
                    resp_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "NepaliRAG",
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(resp_chunk)}\n\n"
                
                # Final stop chunk
                stop_chunk = {
                    "id": req_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                
                duration = round(time.time() - start_time, 2)
                logger.info(f"[{req_id}] RAG Stream finished in {duration}s | Tokens: {in_tokens + count_tokens(full_text)}")

            return StreamingResponse(generate_rag_stream(), media_type="text/event-stream")

        # Non-streaming mode
        full_text = "".join(list(rag.query(user_msg, sys_msg)))
        out_tokens = count_tokens(full_text)
        duration = round(time.time() - start_time, 2)
        
        logger.info(f"[{req_id}] RAG finished in {duration}s | Tokens: {in_tokens} -> {out_tokens}")

        response = {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "NepaliRAG",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": in_tokens,
                "completion_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
                "duration": f"{duration}s"
            }
        }
        return JSONResponse(response)

    except Exception as e:
        logger.error(f"[{req_id}] RAG Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
