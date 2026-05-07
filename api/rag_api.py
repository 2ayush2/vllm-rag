import json
import uuid
import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from utils.logging_utils import setup_logger
from api.dependencies import rag

from schemas.chat import ChatRequest
from utils.token_counter import count_tokens

logger = setup_logger("rag_api_logger", "logs/api.log")

router = APIRouter()

@router.post("/v1/rag/completions")
async def handle_rag_request(req_body: ChatRequest, request: Request):
    req_id = str(uuid.uuid4())
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    # Print full request payload for debugging
    request_data = req_body.model_dump_json(indent=2)
    logger.info(f"\n{'='*50}\n[{req_id}] INCOMING RAG REQUEST:\n{request_data}\n{'='*50}")

    messages = req_body.get_messages()

    user_message = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in input")

    in_tokens = count_tokens(user_message + sys_msg)
    is_stream = req_body.stream

    if is_stream:
        def rag_event_generator():
            accumulated_output = ""
            try:
                for chunk in rag.query(user_message, sys_msg, mode="RAG (PDF Context)"):
                    if chunk:
                        accumulated_output += chunk
                    resp_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "NepaliRAG",
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(resp_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                duration = round(time.time() - start_time, 2)
                out_tokens = count_tokens(accumulated_output)
                logger.info(f"[{req_id}] [{client_ip}] RAG Streaming finished in {duration}s | tokens: {in_tokens}->{out_tokens} (Total: {in_tokens + out_tokens}) | Output preview: {accumulated_output[:300]}")
            except Exception as e:
                logger.error(f"[{req_id}] [{client_ip}] RAG Streaming error: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        return StreamingResponse(rag_event_generator(), media_type="text/event-stream")

    full_response = "".join(list(rag.query(user_message, sys_msg, mode="RAG (PDF Context)")))
    duration = round(time.time() - start_time, 2)

    out_tokens = count_tokens(full_response)
    response = {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "NepaliRAG",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": full_response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "time": f"{duration}s"
        }
    }

    logger.info(f"[{req_id}] [{client_ip}] RAG Response sent in {duration}s | tokens: {in_tokens}->{out_tokens} (Total: {in_tokens + out_tokens}) | Output preview: {full_response[:300]}")
    return JSONResponse(response)
