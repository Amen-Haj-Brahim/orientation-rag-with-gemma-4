import os
from typing import List

import pandas as pd
from langchain_core.documents import Document


def _clean_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_csv_file(file_path: str, document_mode: str = "summary") -> List[Document]:
    documents = []

    if not os.path.exists(file_path):
        print(f"Error: CSV file not found at {file_path}")
        return []

    try:
        print(f"Loading CSV file from: {file_path}")
        df = pd.read_csv(file_path, delimiter=";", encoding="utf-8-sig", dtype=str).fillna("")
        print(f"CSV loaded successfully with {len(df)} rows and {len(df.columns)} columns")

        include_extra_docs = document_mode.lower() in {"full", "expanded", "semantic"}

        for idx, row in df.iterrows():
            row_dict = {key: _clean_value(value) for key, value in row.to_dict().items()}

            row_id = row_dict.get("id", "")
            code = row_dict.get("cod", "")
            cert_ar = row_dict.get("certifar", "")
            cert_fr = row_dict.get("certiffr", "")
            domain = row_dict.get("dom", "")
            university = row_dict.get("univ", "")
            faculty = row_dict.get("etab", "")
            governorate = row_dict.get("gouv", "")
            duration = row_dict.get("duree", "")
            bac = row_dict.get("bac", "")
            score2024 = row_dict.get("score2024", "")
            university_horizon = row_dict.get("horiuniv", "")
            professional_horizon = row_dict.get("horiprof", "")

            summary = f"""
رمز البرنامج: {code}
البرنامج: {cert_ar}
Programme: {cert_fr}
المجال: {domain}
الجامعة: {university}
المؤسسة: {faculty}
الولاية: {governorate}
المدة: {duration}
متطلبات البكالوريا: {bac}
Score 2024: {score2024}
الآفاق الجامعية: {university_horizon}
الآفاق المهنية: {professional_horizon}
"""

            base_metadata = {
                "source_file": os.path.basename(file_path),
                "row_index": idx,
                "row_id": row_id,
                "program_code": code,
                "source_type": "csv",
                "program_ar": cert_ar,
                "program_fr": cert_fr,
                "university": university,
                "faculty": faculty,
                "domain": domain,
                "governorate": governorate,
            }

            documents.append(
                Document(
                    page_content=summary.strip(),
                    metadata={**base_metadata, "doc_type": "summary"},
                )
            )

            if include_extra_docs and cert_ar:
                documents.append(
                    Document(
                        page_content=(
                            f"البرنامج: {cert_ar}\n"
                            f"المؤسسة: {faculty}\n"
                            f"الجامعة: {university}\n"
                            f"المجال: {domain}"
                        ),
                        metadata={**base_metadata, "doc_type": "arabic_certification"},
                    )
                )

            if include_extra_docs and cert_fr:
                documents.append(
                    Document(
                        page_content=(
                            f"Programme: {cert_fr}\n"
                            f"Institution: {faculty}\n"
                            f"Universite: {university}\n"
                            f"Domaine: {domain}"
                        ),
                        metadata={**base_metadata, "doc_type": "french_certification"},
                    )
                )

            if include_extra_docs and professional_horizon:
                documents.append(
                    Document(
                        page_content=(
                            f"الآفاق المهنية: {professional_horizon}\n"
                            f"البرنامج: {cert_ar}"
                        ),
                        metadata={**base_metadata, "doc_type": "professional_horizons"},
                    )
                )

        print(f"Successfully converted {len(df)} CSV rows to {len(documents)} documents")
        return documents

    except Exception as e:
        print(f"Failed to process CSV file: {e}")
        return []


def load_csv_from_directory(directory_path: str, document_mode: str = "summary") -> List[Document]:
    documents = []

    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return []

    csv_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".csv")]
    print(f"Found {len(csv_files)} CSV file(s) in {directory_path}")

    for csv_file in csv_files:
        csv_path = os.path.join(directory_path, csv_file)
        documents.extend(load_csv_file(csv_path, document_mode=document_mode))

    print(f"Total documents extracted: {len(documents)}")
    return documents
