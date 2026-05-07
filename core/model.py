from typing import Any, Optional, Generator
from openai import OpenAI
from core.config import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY

class LocalLLM:
    def __init__(self):
        self.client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        self.model_name = VLLM_MODEL

    def create_response(self, payload: dict) -> Any:
        """Direct call to the unified Responses API."""
        return self.client.responses.create(**payload)

    def generate(self, messages: Any, req_instructions: Optional[str] = None) -> Generator[str, None, None]:
        """Stream generation from the Responses API with the exact logic that works."""
        instructions = req_instructions
        user_messages = messages
        
        if isinstance(messages, list):
            # Extract system message content if present
            sys_msg = next((m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
            if sys_msg:
                instructions = sys_msg
            
            # Filter for non-system messages
            user_messages = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
            
            # Ensure input is never an empty list
            if not user_messages:
                user_messages = "" 
        
        try:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=instructions,
                input=user_messages,
                stream=True,
            )
            for event in response:
                if hasattr(event, "type") and event.type == "response.output_text.delta":
                    if hasattr(event, "delta") and event.delta:
                        yield event.delta
        except Exception as e:
            yield f"LLM Error: {str(e)}"