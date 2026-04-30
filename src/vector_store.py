import os
import time
from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


def create_embeddings_model(api_key: str, model: str = "gemini-embedding-001") -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key
    )


def create_vector_store(
    documents: List[Document],
    embeddings,
    persist_directory: str = "chroma_db",
    collection_name: str = "csv_documents",
    batch_size: int = 50,
    requests_per_minute: int = 60
) -> Chroma:
    os.makedirs(persist_directory, exist_ok=True)
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    if not documents:
        print("Created empty vector store")
        return vector_store

    total = len(documents)
    request_count = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = documents[start:end]

        vector_store.add_documents(batch)
        request_count += 1

        print(f"Embedded documents {end}/{total}")

        if requests_per_minute > 0 and request_count % requests_per_minute == 0 and end < total:
            print("Pausing for 65 seconds to stay under the embedding rate limit...")
            time.sleep(65)
    
    print(f"Created vector store with {len(documents)} documents")
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
