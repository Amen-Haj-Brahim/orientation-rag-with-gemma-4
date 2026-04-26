from src.vector_store import load_vector_store, create_embeddings_model
import os
from dotenv import load_dotenv

def load_config():
    """Load configuration from environment variables."""
    load_dotenv()
    
    return {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "pdf_directory": os.getenv("PDF_DIRECTORY", "data"),
        "chroma_db_dir": os.getenv("CHROMA_DB_DIR", "chroma_db"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        "gemma_model": os.getenv("GEMMA_MODEL", "gemma-4-2b"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200"))
    }







config = load_config()
embeddings = create_embeddings_model(config["api_key"])
vector_store = load_vector_store(config["chroma_db_dir"], embeddings)

# Test query specific to your PDF content
query = "طاقة الاستيعاب كلية العلوم الإنسانية والاجتماعية بتونس" 
results = vector_store.similarity_search(query, k=3)





for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {doc.metadata.get('source_file')}")
    print(f"Content Snippet: {doc.page_content[:500]}")
    
    
# Check how many chunks were actually stored
collection = vector_store._collection
print(f"Total chunks in database: {collection.count()}")


pdf_files = [f for f in os.listdir("./data") if f.lower().endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF files in {"./data"}") # Add this