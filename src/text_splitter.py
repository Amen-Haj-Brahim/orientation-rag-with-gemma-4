from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "، ", "؛ ", " ", ""],
        add_start_index=True,
        strip_whitespace=True,
    )


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def normalize_arabic_text(text: str) -> str:
    arabic_diacritics = [
        "\u064B",
        "\u064C",
        "\u064D",
        "\u064E",
        "\u064F",
        "\u0650",
        "\u0651",
        "\u0652",
        "\u0670",
    ]
    for diacritic in arabic_diacritics:
        text = text.replace(diacritic, "")

    replacements = {
        "\u0622": "\u0627",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0624": "\u0648",
        "\u0626": "\u064A",
        "\u0640": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    return text
