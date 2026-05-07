import json, uuid, time
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
        input_data = req.get_input()
        model_name = req.model
        if not model_name:
            raise HTTPException(status_code=400, detail="model is required in the request.")
        instructions = req.instructions
        is_stream = req.stream
        logger.info(f"[{req_id}] Incoming request: {req.model_dump_json(indent=2)}")
        in_tokens = count_tokens(json.dumps(input_data))
        if is_stream:
            def generate_stream():
                full_text = ""
                for chunk in llm.generate(input_data, req_instructions=instructions):
                    if not chunk: continue
                    full_text += chunk
                    
                    # Standard Response API chunk
                    resp = {
                        "id": req_id,
                        "object": "response.chunk",
                        "type": "response.output_text.delta",
                        "delta": chunk,
                    }
                    yield f"data: {json.dumps(resp, ensure_ascii=False)}\n\n"
                
                # Final completion event
                yield f"data: {json.dumps({'id': req_id, 'object': 'response', 'type': 'response.completed'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                
                duration = round(time.time() - start_time, 2)
                logger.info(f"[{req_id}] Stream finished in {duration}s | Tokens: {in_tokens + count_tokens(full_text)}")

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # Non-streaming mode
        full_text = "".join(llm.generate(input_data, req_instructions=instructions))
        out_tokens = count_tokens(full_text)
        duration = round(time.time() - start_time, 2)
        
        logger.info(f"[{req_id}] Finished in {duration}s | Tokens: {in_tokens} -> {out_tokens}")

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

    except Exception as e:
        logger.error(f"[{req_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))