"""
Retrieval Module

Handles document retrieval based on user queries.
"""

from typing import List
from langchain_core.documents import Document


def retrieve_documents(retriever, query: str, k: int = 4) -> List[Document]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        retriever: Retriever instance
        query: User query
        k: Number of documents to retrieve
        
    Returns:
        List of relevant Document objects
    """
    docs = retriever.get_relevant_documents(query)
    return docs[:k]


def format_retrieved_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents for display.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Formatted string
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        formatted.append(f"--- Document {i} (Source: {source}, Page: {page}) ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_sources(docs: List[Document]) -> List[str]:
    """
    Get unique source files from retrieved documents.
    
    Args:
        docs: List of Document objects
        
    Returns:
        List of source file names
    """
    sources = set()
    for doc in docs:
        source = doc.metadata.get('source_file')
        if source:
            sources.add(source)
    return sorted(sources)