import os
import shutil
import time

import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.model import LocalLLM



DB_DIR = "./database/chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class NepaliRAG:
    def __init__(self):
        self.llm = LocalLLM()
        self.embeddings = self._load_embeddings()
        self.db = None
        self.current_doc = None
        self._try_load_existing_db()

    # ── Setup ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_embeddings() -> HuggingFaceEmbeddings:
        print(f"Loading embedding model ({EMBEDDING_MODEL}) on cpu...")
        os.environ["PYTORCH_cpu_ALLOC_CONF"] = "expandable_segments:True"
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
        )
        print("Embedding model loaded.")
        return model

    def _try_load_existing_db(self) -> None:
        if not (os.path.exists(DB_DIR) and os.listdir(DB_DIR)):
            return
        try:
            print("Found existing database. Loading...")
            self.db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
            self.db.similarity_search("query: test", k=1)
            self.current_doc = "Previous Document"
            print("Existing database loaded successfully.")
        except Exception as e:
            print(f"Database error ({e}). Removing stale data.")
            self.db = None
            self._clear_db()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clear_db(self) -> None:
        """Release the Chroma client and wipe the DB directory."""
        if self.db is not None:
            try:
                self.db._client.stop()
            except Exception:
                pass
            self.db = None

        time.sleep(1)

        if os.path.exists(DB_DIR):
            for attempt in range(3):
                try:
                    shutil.rmtree(DB_DIR)
                    print(f"Cleared database at {DB_DIR}")
                    return
                except Exception as e:
                    print(f"Retry {attempt + 1}: Could not delete {DB_DIR}: {e}")
                    time.sleep(1)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_document(self, pdf_path: str) -> str:
        filename = os.path.basename(pdf_path)

        if self.current_doc == filename and self.db is not None:
            return f"Document **{filename}** is already indexed and ready."

        try:
            print(f"Indexing: {filename}  |  model: {EMBEDDING_MODEL}")

            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50,
            ).split_documents(PyMuPDFLoader(pdf_path).load())

            if not chunks:
                return "The PDF appears to be empty or unreadable."

            # e5 models need a "passage: " prefix at index time
            passages = [
                Document(page_content="passage: " + d.page_content, metadata=d.metadata)
                for d in chunks
            ]

            self._clear_db()

            if torch.cpu.is_available():
                torch.cpu.empty_cache()

            self.db = Chroma.from_documents(
                documents=passages,
                embedding=self.embeddings,
                persist_directory=DB_DIR,
            )
            self.current_doc = filename
            return f"Document **{filename}** indexed successfully ({len(passages)} chunks)."

        except Exception as e:
            print(f"Ingestion error: {e}")
            return f"Failed to load document: {e}"

    def query(self, user_input: str, sys_msg: str = None, mode: str = "RAG (PDF Context)"):
        is_rag = mode == "RAG (PDF Context)"

        if is_rag and self.db is None:
            yield "Please upload a PDF document first using the panel on the left."
            return

        try:
            messages = []
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            
            if is_rag:
                results = self.db.similarity_search("query: " + user_input, k=4)
                if not results:
                    yield "No relevant context found in the document."
                    return
                context = "\n---\n".join(
                    d.page_content.removeprefix("passage: ") for d in results
                )
                prompt = f"Context:\n{context}\n\nQuestion:\n{user_input}\n\nAnswer:"
            else:
                prompt = user_input

            messages.append({"role": "user", "content": prompt})
            
            yield from self.llm.generate(messages)


        except Exception as e:
            if "dimension" in str(e).lower():
                yield "Error: Embedding dimension mismatch — please re-upload your document."
            else:
                yield f"Query error: {e}"