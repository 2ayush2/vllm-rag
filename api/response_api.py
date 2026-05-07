import json
import uuid
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from schemas.chat import ResponseRequest, ResponseAPIResponse, ResponseOutputMessage, ResponseOutputText, ResponseUsage
from api.dependencies import llm
from utils.logging_utils import setup_logger
from utils.token_counter import count_tokens

logger = setup_logger("api_logger", "logs/api.log")
router = APIRouter()


@router.post("/v1/responses")
async def responses_api(req: ResponseRequest):
    req_id = f"resp_{uuid.uuid4().hex}"
    start_time = time.time()
    try:
        request_data = req.model_dump_json(indent=2)
        logger.info(f"\n{'='*50}\n[{req_id}] INCOMING REQUEST:\n{request_data}\n{'='*50}")
        input_data = req.get_input()
        stream = req.stream
        instructions = req.instructions
        model_name = req.model
        if not model_name:
            raise HTTPException(status_code=400, detail="model is required in the request.")
        in_tokens = count_tokens(json.dumps(input_data, ensure_ascii=False))
        def generate():
            full_text = ""
            for chunk in llm.generate(input_data, req_instructions=instructions):
                if not chunk:
                    continue
                full_text += chunk
                if stream:
                    resp = {
                        "id": req_id,
                        "object": "response.chunk",
                        "type": "response.output_text.delta",
                        "delta": chunk,
                    }
                    yield f"data: {json.dumps(resp, ensure_ascii=False)}\n\n"
            if stream:
                completed_event = {
                    "id": req_id,
                    "object": "response",
                    "type": "response.completed",
                }
                yield f"data: {json.dumps(completed_event, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            duration = round(time.time() - start_time, 2)
            logger.info(
                f"[{req_id}] stream done in {duration}s | "
                f"total_tokens: {in_tokens + count_tokens(full_text)} | "
                f"output={full_text[:200]}..."
            )
        if stream:
            return StreamingResponse(generate(), media_type="text/event-stream")
        full_text = "".join(llm.generate(input_data, req_instructions=instructions))
        out_tokens = count_tokens(full_text)
        duration = round(time.time() - start_time, 2)
        logger.info(
            f"[{req_id}] response done in {duration}s | "
            f"tokens: {in_tokens}->{out_tokens} "
            f"(Total: {in_tokens + out_tokens}) | "
            f"output={full_text[:200]}..."
        )
        response_data = ResponseAPIResponse(
            id=req_id,
            created_at=int(time.time()),
            model=model_name,
            output=[
                ResponseOutputMessage(
                    id=f"msg_{uuid.uuid4().hex}",
                    content=[ResponseOutputText(text=full_text)]
                )
            ],
            usage=ResponseUsage(
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                total_tokens=in_tokens + out_tokens
            )
        )
        return JSONResponse(content=response_data.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))