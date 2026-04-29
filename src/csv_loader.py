import os
import pandas as pd
from typing import List
from langchain_core.documents import Document


def _clean_value(value) -> str:
    """Return a clean string for CSV values, including empty cells."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_csv_file(file_path: str, document_mode: str = "summary") -> List[Document]:
    """
    Loads a CSV file and converts each row into semantic LangChain Documents.
    Creates multiple documents per row for better retrieval.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    if not os.path.exists(file_path):
        print(f"Error: CSV file not found at {file_path}")
        return []
    
    try:
        print(f"Loading CSV file from: {file_path}")
        
        # Read CSV with semicolon delimiter. utf-8-sig handles a possible BOM.
        df = pd.read_csv(file_path, delimiter=";", encoding="utf-8-sig", dtype=str).fillna("")
        print(f"CSV loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        
        document_mode = document_mode.lower()
        include_extra_docs = document_mode in {"full", "expanded", "semantic"}

        # Convert each row into one rich document by default. Extra focused
        # documents are useful, but they cost more embedding quota.
        for idx, row in df.iterrows():
            # Create a comprehensive text representation of the row
            row_dict = {key: _clean_value(value) for key, value in row.to_dict().items()}
            
            # Document 1: Summary document with all info in natural language
            # Extract key fields
            row_id = row_dict.get('id', '')
            code = row_dict.get('cod', '')
            cert_ar = row_dict.get('certifar', '')  # Arabic certification
            cert_fr = row_dict.get('certiffr', '')  # French certification
            dom = row_dict.get('dom', '')  # Domain
            univ = row_dict.get('univ', '')  # University
            etab = row_dict.get('etab', '')  # Faculty/Establishment
            gouv = row_dict.get('gouv', '')  # Governorate
            duree = row_dict.get('duree', '')  # Duration
            bac = row_dict.get('bac', '')  # Baccalaureate requirement
            score2024 = row_dict.get('score2024', '')
            horiz_univ = row_dict.get('horiuniv', '')  # University horizon
            horiz_prof = row_dict.get('horiprof', '')  # Professional horizon
            
            # Build comprehensive summary
            summary = f"""
رمز البرنامج: {code}
البرنامج: {cert_ar}
Programme: {cert_fr}
المجال: {dom}
الجامعة: {univ}
المؤسسة: {etab}
الولاية: {gouv}
المدة: {duree}
متطلبات البكالوريا: {bac}
Score 2024: {score2024}
الآفاق الجامعية: {horiz_univ}
الآفاق المهنية: {horiz_prof}
"""
            
            # Create main document
            main_doc = Document(
                page_content=summary.strip(),
                metadata={
                    "source_file": os.path.basename(file_path),
                    "row_index": idx,
                    "row_id": row_id,
                    "program_code": code,
                    "source_type": "csv",
                    "doc_type": "summary",
                    "program_ar": cert_ar,
                    "program_fr": cert_fr,
                    "university": univ,
                    "faculty": etab,
                    "domain": dom,
                    "governorate": gouv
                }
            )
            documents.append(main_doc)
            
            # Optional focused documents for stronger retrieval at higher quota cost.
            if include_extra_docs and cert_ar:
                doc_ar = Document(
                    page_content=f"البرنامج: {cert_ar}\nالمؤسسة: {etab}\nالجامعة: {univ}\nالمجال: {dom}",
                    metadata={
                        "source_file": os.path.basename(file_path),
                        "row_index": idx,
                        "row_id": row_id,
                        "program_code": code,
                        "source_type": "csv",
                        "doc_type": "arabic_certification",
                        "program_ar": cert_ar,
                        "university": univ
                    }
                )
                documents.append(doc_ar)
            
            if include_extra_docs and cert_fr:
                doc_fr = Document(
                    page_content=f"Programme: {cert_fr}\nInstitution: {etab}\nUniversite: {univ}\nDomaine: {dom}",
                    metadata={
                        "source_file": os.path.basename(file_path),
                        "row_index": idx,
                        "row_id": row_id,
                        "program_code": code,
                        "source_type": "csv",
                        "doc_type": "french_certification",
                        "program_fr": cert_fr,
                        "university": univ
                    }
                )
                documents.append(doc_fr)
            
            if include_extra_docs and horiz_prof:
                doc_prof = Document(
                    page_content=f"الآفاق المهنية: {horiz_prof}\nالبرنامج: {cert_ar}",
                    metadata={
                        "source_file": os.path.basename(file_path),
                        "row_index": idx,
                        "row_id": row_id,
                        "program_code": code,
                        "source_type": "csv",
                        "doc_type": "professional_horizons",
                        "program_ar": cert_ar
                    }
                )
                documents.append(doc_prof)
        
        print(f"Successfully converted {len(df)} CSV rows to {len(documents)} semantic documents")
        return documents
        
    except Exception as e:
        print(f"Failed to process CSV file: {str(e)}")
        return []


def load_csv_from_directory(directory_path: str, document_mode: str = "summary") -> List[Document]:
    """
    Loads all CSV files from a directory and converts them into Documents.
    
    Args:
        directory_path: Path to directory containing CSV files
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return []
    
    csv_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.csv')]
    print(f"Found {len(csv_files)} CSV file(s) in {directory_path}")
    
    for csv_file in csv_files:
        csv_path = os.path.join(directory_path, csv_file)
        loaded_docs = load_csv_file(csv_path, document_mode=document_mode)
        documents.extend(loaded_docs)
    
    print(f"Total documents extracted: {len(documents)}")
    return documents
