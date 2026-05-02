import os
import time
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers or set EMBEDDING_PROVIDER=gemini."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.uses_e5_prefix = "e5" in model_name.lower()

    def _prepare_documents(self, texts: List[str]) -> List[str]:
        if not self.uses_e5_prefix:
            return texts
        return [f"passage: {text}" for text in texts]

    def _prepare_query(self, text: str) -> str:
        if not self.uses_e5_prefix:
            return text
        return f"query: {text}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            self._prepare_documents(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            self._prepare_query(text),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


def create_embeddings_model(
    api_key: str,
    model: str = "gemini-embedding-001",
    provider: str = "gemini",
    local_model: str = "intfloat/multilingual-e5-small",
    local_device: str = "cpu",
):
    if provider.lower() == "local":
        print(f"Using local embeddings: {local_model}")
        return LocalSentenceTransformerEmbeddings(local_model, device=local_device)

    print(f"Using Gemini embeddings: {model}")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def create_vector_store(
    documents: List[Document],
    embeddings,
    persist_directory: str = "chroma_db",
    collection_name: str = "csv_documents",
    batch_size: int = 20,
) -> Chroma:
    os.makedirs(persist_directory, exist_ok=True)
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    if not documents:
        return vector_store

    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = documents[start:end]

        # Retry logic loop
        while True:
            try:
                vector_store.add_documents(batch)
                print(f"Embedded documents {end}/{total}")
                # Short break between batches to stay under the RPM
                time.sleep(2) 
                break 
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"Quota exceeded. Sleeping for 60s...")
                    time.sleep(60)
                else:
                    print(f"Unexpected error: {e}")
                    raise e
    
    return vector_store


def load_vector_store(
    persist_directory: str,
    embeddings,
    collection_name: str = "csv_documents"
) -> Optional[Chroma]:
    if not os.path.exists(persist_directory):
        return None
    
    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


def get_retriever(vector_store: Chroma, search_kwargs: Optional[dict] = None):
    if search_kwargs is None:
        search_kwargs = {"k": 4}
    
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
