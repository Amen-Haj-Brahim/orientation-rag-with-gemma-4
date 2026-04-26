"""
Text Splitter Module

Splits documents into chunks for vector storage.
Includes Arabic text handling.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with specified parameters.
    
    Args:
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "، ",   # Arabic comma
            "؛ ",   # Arabic semicolon
            " ",     # Words
            ""       # Characters
        ],
        add_start_index=True,
        strip_whitespace=True
    )


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for better processing.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Remove diacritics (tashkeel)
    arabic_diacritics = ['\u064B', '\u064C', '\u064D', '\u064E', '\u064F', 
                         '\u0650', '\u0651', '\u0652', '\u0670']
    for diacritic in arabic_diacritics:
        text = text.replace(diacritic, '')
    
    # Normalize alef variants
    text = text.replace('\u0622', '\u0627')  # Alef with madda -> alef
    text = text.replace('\u0623', '\u0627')  # Alef with hamza -> alef
    text = text.replace('\u0625', '\u0627')  # Alef with hamza below -> alef
    text = text.replace('\u0624', '\u0648')  # Waw with hamza -> waw
    text = text.replace('\u0626', '\u064A')  # Yeh with hamza -> yeh
    
    # Remove tatweel (kashida)
    text = text.replace('\u0640', '')
    
    return text