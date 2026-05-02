import os
import re
import unicodedata
from typing import List

import fitz
from langchain_core.documents import Document


def clean_pdf_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.isdigit() and len(line) <= 3:
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def is_useful_page(text: str, min_chars: int = 120) -> bool:
    if len(text) < min_chars:
        return False

    letters = sum(1 for char in text if char.isalpha())
    return letters >= min_chars // 2


def load_pdf_file(file_path: str, min_chars: int = 120) -> List[Document]:
    documents = []

    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return documents

    print(f"Loading PDF file from: {file_path}")
    pdf = fitz.open(file_path)

    for page_index, page in enumerate(pdf):
        text = clean_pdf_text(page.get_text("text"))
        if not is_useful_page(text, min_chars=min_chars):
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_file": os.path.basename(file_path),
                    "source_type": "pdf",
                    "doc_type": "pdf_page",
                    "page": page_index + 1,
                },
            )
        )

    print(f"Extracted {len(documents)} useful pages from {os.path.basename(file_path)}")
    return documents


def load_pdfs_from_directory(directory_path: str, min_chars: int = 120) -> List[Document]:
    documents = []

    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return documents

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF file(s) in {directory_path}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        documents.extend(load_pdf_file(pdf_path, min_chars=min_chars))

    print(f"Total PDF pages extracted: {len(documents)}")
    return documents
