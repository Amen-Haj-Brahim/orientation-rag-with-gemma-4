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
    docs = retriever.invoke(query)
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
        row_id = doc.metadata.get('row_id', doc.metadata.get('row_index', 'N/A'))
        code = doc.metadata.get('program_code', 'N/A')
        formatted.append(f"--- Document {i} (Source: {source}, Row: {row_id}, Code: {code}) ---\n{doc.page_content}")
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
