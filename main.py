"""
Main Entry Point

RAG pipeline for querying CSV data using Gemma.
"""

import os
import shutil
import sys
from dotenv import load_dotenv
from src.csv_loader import load_csv_from_directory
from src.text_splitter import split_documents, normalize_arabic_text
from src.vector_store import create_embeddings_model, create_vector_store, load_vector_store, get_retriever
from src.rag_pipeline import create_gemma_llm, create_rag_pipeline, answer_question, format_answer


def configure_console():
    """Use UTF-8 output when the host console supports reconfiguration."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def load_config():
    """Load configuration from environment variables."""
    load_dotenv()
    
    csv_directory = os.getenv("CSV_DIRECTORY") or os.getenv("PDF_DIRECTORY", "data")

    return {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "csv_directory": csv_directory,
        "chroma_db_dir": os.getenv("CHROMA_DB_DIR", "chroma_db"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        "gemma_model": os.getenv("GEMMA_MODEL", "gemma-4-2b"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "3000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "500")),
        "csv_document_mode": os.getenv("CSV_DOCUMENT_MODE", "summary"),
        "embedding_batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "50")),
        "embedding_requests_per_minute": int(os.getenv("EMBEDDING_REQUESTS_PER_MINUTE", "60"))
    }


def initialize_rag(config: dict, force_reindex: bool = False):
    """
    Initialize the RAG pipeline.
    
    Args:
        config: Configuration dictionary
        force_reindex: Force re-indexing of documents
        
    Returns:
        Tuple of (rag_pipeline, vector_store)
    """
    api_key = config["api_key"]
    
    if not api_key or api_key == "your_api_key_here":
        print("Error: Please set your GOOGLE_API_KEY in the .env file")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    # Create embeddings model
    print("Initializing embeddings model...")
    embeddings = create_embeddings_model(api_key, config["embedding_model"])
    
    # Try to load existing vector store
    vector_store = None
    if force_reindex and os.path.exists(config["chroma_db_dir"]):
        print("Deleting old vector store...")
        shutil.rmtree(config["chroma_db_dir"])

    if not force_reindex:
        print("Loading existing vector store...")
        vector_store = load_vector_store(config["chroma_db_dir"], embeddings, "csv_documents")
    
    # Create new vector store if needed
    if vector_store is None:
        print("Creating new vector store...")
        
        # Load CSV files
        print(f"Loading CSV files from: {config['csv_directory']}")
        print(f"CSV document mode: {config['csv_document_mode']}")
        try:
            documents = load_csv_from_directory(
                config["csv_directory"],
                document_mode=config["csv_document_mode"]
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please create the data directory and add CSV files.")
            sys.exit(1)

        if not documents:
            print("Error: No CSV documents were loaded. Check CSV_DIRECTORY and file encoding.")
            sys.exit(1)
        
        # Normalize Arabic text
        print("Normalizing Arabic text...")
        for doc in documents:
            doc.page_content = normalize_arabic_text(doc.page_content)
        
        # Split documents
        print("Splitting documents into chunks...")
        chunks = split_documents(
            documents,
            config["chunk_size"],
            config["chunk_overlap"]
        )
        
        # Create vector store
        print("Creating vector store...")
        vector_store = create_vector_store(
            chunks,
            embeddings,
            config["chroma_db_dir"],
            collection_name="csv_documents",
            batch_size=config["embedding_batch_size"],
            requests_per_minute=config["embedding_requests_per_minute"]
        )
    
    # Create retriever
    print("Creating retriever...")
    retriever = get_retriever(vector_store, {"k": 10})
    
    # Create Gemma LLM
    print("Initializing Gemma model...")
    llm = create_gemma_llm(api_key, config["gemma_model"])
    
    # Create RAG pipeline
    print("Creating RAG pipeline...")
    rag_pipeline = create_rag_pipeline(llm, retriever)
    
    return rag_pipeline, vector_store


def main():
    """Main function."""
    configure_console()

    print("=" * 60)
    print("RAG with Gemma - CSV Question Answering")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Check for --reindex flag
    force_reindex = "--reindex" in sys.argv
    
    # Initialize RAG pipeline
    rag_pipeline, vector_store = initialize_rag(config, force_reindex)
    
    print("\n" + "=" * 60)
    print("Ready! Enter your questions (or 'quit' to exit)")
    print("=" * 60)
    
    # Interactive loop
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            question = normalize_arabic_text(question)
            
            # Get answer
            print("\nThinking...")
            
            result = answer_question(rag_pipeline, question)
            
            # Display answer
            print("\n" + format_answer(result))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
