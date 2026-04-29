import os
import shutil
import sys

from dotenv import load_dotenv

from src.csv_loader import load_csv_from_directory
from src.text_splitter import normalize_arabic_text, split_documents
from src.vector_store import create_embeddings_model, create_vector_store


def configure_console():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def load_config():
    load_dotenv()

    return {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "data_directory": os.getenv("CSV_DIRECTORY") or os.getenv("PDF_DIRECTORY", "data"),
        "chroma_db_dir": os.getenv("CHROMA_DB_DIR", "chroma_db"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "3000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "500")),
        "csv_document_mode": os.getenv("CSV_DOCUMENT_MODE", "summary"),
        "embedding_batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "50")),
        "embedding_requests_per_minute": int(os.getenv("EMBEDDING_REQUESTS_PER_MINUTE", "60")),
    }


def main():
    configure_console()
    config = load_config()

    if not config["api_key"] or config["api_key"] == "your_google_api_key_here":
        print("Set GOOGLE_API_KEY in .env before running retrieval tests.")
        return

    if os.path.exists(config["chroma_db_dir"]):
        print("Deleting old vector store...")
        shutil.rmtree(config["chroma_db_dir"])

    print("Creating embeddings model...")
    embeddings = create_embeddings_model(config["api_key"], config["embedding_model"])

    print("Loading CSV data...")
    documents = load_csv_from_directory(
        config["data_directory"],
        document_mode=config["csv_document_mode"],
    )
    print(f"Loaded {len(documents)} documents")

    if not documents:
        print("No documents loaded. Check CSV_DIRECTORY and CSV files.")
        return

    print("\n--- Sample Documents ---")
    for i, document in enumerate(documents[:3], 1):
        print(f"\nDocument {i}:")
        print(f"Content: {document.page_content[:500]}")
        print(f"Metadata: {document.metadata}")

    print("\nNormalizing Arabic text...")
    for document in documents:
        document.page_content = normalize_arabic_text(document.page_content)

    print("Splitting documents into chunks...")
    chunks = split_documents(
        documents,
        config["chunk_size"],
        config["chunk_overlap"],
    )

    print("Creating vector store...")
    vector_store = create_vector_store(
        chunks,
        embeddings,
        config["chroma_db_dir"],
        collection_name="csv_documents",
        batch_size=config["embedding_batch_size"],
        requests_per_minute=config["embedding_requests_per_minute"],
    )

    print("\n" + "=" * 60)
    print("Testing Retrieval")
    print("=" * 60)

    test_queries = [
        "برامج العربية",
        "جامعة تونس",
        "License en Arabe",
        "آفاق مهنية",
        "كلية العلوم الإنسانية",
    ]

    for query in test_queries:
        normalized_query = normalize_arabic_text(query)
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search(normalized_query, k=3)

        if not results:
            print("  No results found")
            continue

        for i, document in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Content: {document.page_content[:300]}")
            print(f"    Metadata: {document.metadata}")

    csv_files = [f for f in os.listdir(config["data_directory"]) if f.lower().endswith(".csv")]
    print(f"\nFound {len(csv_files)} CSV files in {config['data_directory']}")


if __name__ == "__main__":
    main()
