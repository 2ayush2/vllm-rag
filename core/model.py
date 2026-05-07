from openai import OpenAI

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "./models/qwen3-4b-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
class LocalLLM:
    def __init__(self):
        self.client = OpenAI(base_url=VLLM_BASE_URL, api_key="token-abc123")
        self.model_name = VLLM_MODEL

    def create_response(self, payload: dict):
        """Direct call to the unified Responses API."""
        return self.client.responses.create(**payload)

    def generate(self, messages, req_instructions=None):
        """Directly stream generation from the Responses API."""
        instructions = req_instructions
        user_messages = messages
        
        if isinstance(messages, list):
            sys_msg = next((m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
            if sys_msg:
                instructions = sys_msg
            user_messages = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
            if not user_messages:
                user_messages = "" # Fallback if empty
        
        try:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=instructions,
                input=user_messages,
                stream=True,
            )
            for event in response:
                # The Responses API returns events, not chunks with choices
                if hasattr(event, "type") and event.type == "response.output_text.delta":
                    if hasattr(event, "delta") and event.delta:
                        yield event.delta
        except Exception as e:
            yield f"LLM Error: {str(e)}"