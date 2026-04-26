"""
Main Entry Point

RAG pipeline for querying PDF documents using Gemma.
"""

import os
import sys
from dotenv import load_dotenv
from src.pdf_loader import load_pdfs_from_directory
from src.text_splitter import split_documents, normalize_arabic_text
from src.vector_store import create_embeddings_model, create_vector_store, load_vector_store, get_retriever
from src.rag_pipeline import create_gemma_llm, create_rag_pipeline, answer_question, format_answer
from src.retrieval import get_sources


os.environ["HF_TOKEN"] = "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"

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
    if not force_reindex:
        print("Loading existing vector store...")
        vector_store = load_vector_store(config["chroma_db_dir"], embeddings)
    
    # Create new vector store if needed
    if vector_store is None:
        print("Creating new vector store...")
        
        # Load PDFs
        print(f"Loading PDFs from: {config['pdf_directory']}")
        try:
            documents = load_pdfs_from_directory(config["pdf_directory"])
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please create the data directory and add PDF files.")
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
            config["chroma_db_dir"]
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
    print("=" * 60)
    print("RAG with Gemma - PDF Question Answering")
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