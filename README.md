# vLLM RAG API 🚀

A high-performance, strictly-formatted Retrieval-Augmented Generation (RAG) backend powered by **FastAPI** and **vLLM**. 

This project utilizes the newest **OpenAI Responses API (`/v1/responses`)** format. By explicitly decoupling `instructions` (System Rules) from `input` (User Context & Queries), this API forces the underlying LLM to strictly adhere to complex, bilingual (English/Nepali) formatting rules, completely avoiding "Lost in the Middle" or prompt overshadowing issues common in massive nested context RAGs.

## Features ✨
- **Responses API Integration**: Streams responses via `ResponseStreamEvent`, enforcing strict priority on system instructions over payload data.
- **Bilingual Support**: Configured to seamlessly handle contextual responses in both **Nepali** and **English**.
- **Nested Payload Parsing**: Robust Pydantic sub-models specifically handle deep, nested JSON containing RAG context, URLs, system prompts, and queries.
- **Streaming & Non-Streaming Modes**: Supports robust `text/event-stream` generation or blocking JSON responses.
- **Powered by Qwen3**: Optimized for the `Qwen3-4B-Instruct` GGUF model served locally via vLLM.

---

## 🛠 Prerequisites
- Python 3.10+
- [vLLM](https://github.com/vllm-project/vllm) for local model serving.
- A GGUF model downloaded to `./models/qwen3-4b-gguf/`

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:2ayush2/vllm-rag.git
   cd vllm-rag
   ```

2. **Setup Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt # Make sure fastapi, uvicorn, openai, pydantic are installed
   ```

3. **Start the vLLM Server (Terminal 1):**
   ```bash
   vllm serve "./models/qwen3-4b-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" --port 8000
   ```

4. **Start the FastAPI Application (Terminal 2):**
   ```bash
   python main.py
   ```
   *The API will be available at `http://0.0.0.0:8001`*

---

## 📖 API Usage

**Endpoint**: `POST /v1/responses`

This endpoint strictly follows the new Responses API signature. Send your nested JSON logic inside the `input` string, and your strict rules inside `instructions`.

```json
{
  "model": "qwen3-rag",
  "instructions": "You are a bilingual RAG assistant. Only output in JSON format.",
  "input": "{\"system_prompt\": \"...\", \"context\": [...], \"query\": \"मलाई नाबिल बैंकको बारेमा भन्नुहोस्\"}",
  "stream": true
}
```

---

## 📂 Project Structure
- `api/` - FastAPI routing controllers (`response_api.py`, `rag_api.py`)
- `core/` - Core LLM engine connection (`model.py`) connecting to vLLM via the OpenAI Client.
- `schemas/` - Pydantic definitions for safe Request/Response handling (`chat.py`).
- `utils/` - Utilities for logging and token counting.
- `main.py` - Uvicorn entry point.

---
*Built with ❤️ for precision bilingual RAG.*
