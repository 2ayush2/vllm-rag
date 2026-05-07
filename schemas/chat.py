from pydantic import BaseModel
from typing import List, Optional, Union, Dict
import json

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Union[Dict[str, str], Message]]
    stream: bool = False

    def get_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": m.role, "content": m.content} if isinstance(m, Message) else m
            for m in self.messages
        ]

class RAGInputPayload(BaseModel):
    system_prompt: Union[Dict, str] = ""
    context: List = []
    query: str = ""

    def to_messages(self) -> List[Dict[str, str]]:
        sys_prompt_str = json.dumps(self.system_prompt, ensure_ascii=False, indent=2) if isinstance(self.system_prompt, dict) else str(self.system_prompt)
        user_content = self.query
        if self.context:
            user_content = f"Context:\n{json.dumps(self.context, ensure_ascii=False, indent=2)}\n\nQuery:\n{self.query}"
        messages = []
        if sys_prompt_str.strip():
            messages.append({"role": "system", "content": sys_prompt_str})
        messages.append({"role": "user", "content": user_content})
        return messages

class ResponseRequest(BaseModel):
    model: Optional[str] = None
    input: Optional[Union[str, List[Message], List[Dict[str, str]]]] = None
    messages: Optional[Union[str, List[Message], List[Dict[str, str]]]] = None
    instructions: Optional[str] = None
    stream: bool = True

    def get_input(self) -> Union[str, List[Dict[str, str]]]:
        raw_input = self.input or self.messages
        if isinstance(raw_input, str):
            try:
                parsed_payload = RAGInputPayload.model_validate_json(raw_input)
                if parsed_payload.query:
                    return parsed_payload.to_messages()
            except Exception: pass
            return raw_input
        if not raw_input: return ""
        return [{"role": msg.role, "content": msg.content} if isinstance(msg, Message) else msg for msg in raw_input]

class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class ResponseOutputText(BaseModel):
    type: str = "output_text"
    text: str

class ResponseOutputMessage(BaseModel):
    id: str
    type: str = "message"
    status: str = "completed"
    role: str = "assistant"
    content: List[ResponseOutputText]

class ResponseAPIResponse(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    model: str
    output: List[ResponseOutputMessage]
    usage: ResponseUsage