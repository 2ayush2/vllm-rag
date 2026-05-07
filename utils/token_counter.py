import tiktoken

def count_tokens(text: str) -> int:
    """
    Fast estimation of token count using tiktoken.
    This is an approximation for Qwen models but very fast.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimate: ~4 chars per token
        return len(text) // 4
