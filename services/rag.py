import os
import shutil
import time
from typing import List, Optional, Generator

import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.model import LocalLLM
from core.config import DB_DIR, EMBEDDING_MODEL

class NepaliRAG:
    def __init__(self):
        self.llm = LocalLLM()
        self.embeddings = self._load_embeddings()
        self.db = None
        self.current_doc = None
        self._try_load_existing_db()

    @staticmethod
    def _load_embeddings() -> HuggingFaceEmbeddings:
        """Initialize embedding model with CPU optimizations."""
        os.environ["PYTORCH_cpu_ALLOC_CONF"] = "expandable_segments:True"
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
        )

    def _try_load_existing_db(self) -> None:
        """Attempts to load a previously persisted Chroma database."""
        if not (os.path.exists(DB_DIR) and os.listdir(DB_DIR)):
            return
        try:
            self.db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
            self.db.similarity_search("query: test", k=1)
            self.current_doc = "Previous Document"
        except Exception:
            self._clear_db()

    def _clear_db(self) -> None:
        """Safely wipes the local Chroma database directory."""
        if self.db is not None:
            try:
                if hasattr(self.db, "_client") and hasattr(self.db._client, "stop"):
                    self.db._client.stop()
            except Exception: pass
            self.db = None

        time.sleep(1)
        if os.path.exists(DB_DIR):
            for _ in range(3):
                try:
                    shutil.rmtree(DB_DIR)
                    break
                except Exception: time.sleep(1)

    def _build_rag_prompt(self, user_input: str, results: List[Document]) -> str:
        """Constructs a clean prompt with document context."""
        context = "\n---\n".join(d.page_content.removeprefix("passage: ") for d in results)
        return (
            "You are a helpful assistant. Use the following context to answer the user's question.\n"
            "If the answer is not in the context, say you don't know based on the provided document.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_input}\n\n"
            "Answer:"
        )

    def load_document(self, pdf_path: str) -> str:
        """Indexes a PDF document into the local Chroma vector store."""
        filename = os.path.basename(pdf_path)
        if self.current_doc == filename and self.db is not None:
            return f"Document **{filename}** is already ready."

        try:
            # Load and split
            loader = PyMuPDFLoader(pdf_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(loader.load())

            if not chunks:
                return "The PDF appears to be empty."

            # Prefix for e5 models
            passages = [Document(page_content="passage: " + d.page_content, metadata=d.metadata) for d in chunks]

            self._clear_db()
            if hasattr(torch, "cpu") and hasattr(torch.cpu, "empty_cache"):
                torch.cpu.empty_cache()

            self.db = Chroma.from_documents(documents=passages, embedding=self.embeddings, persist_directory=DB_DIR)
            self.current_doc = filename
            return f"Document **{filename}** indexed successfully."
        except Exception as e:
            return f"Failed to load document: {str(e)}"

    def query(self, user_input: str, sys_msg: Optional[str] = None, mode: str = "RAG (PDF Context)") -> Generator[str, None, None]:
        """Main query interface for RAG and standard chat."""
        if mode == "RAG (PDF Context)" and self.db is None:
            yield "Please upload a PDF document first."
            return

        try:
            messages = [{"role": "system", "content": sys_msg}] if sys_msg else []
            
            if mode == "RAG (PDF Context)":
                results = self.db.similarity_search("query: " + user_input, k=4)
                prompt = self._build_rag_prompt(user_input, results) if results else user_input
            else:
                prompt = user_input

            messages.append({"role": "user", "content": prompt})
            yield from self.llm.generate(messages)
        except Exception as e:
            yield f"Query error: {str(e)}"