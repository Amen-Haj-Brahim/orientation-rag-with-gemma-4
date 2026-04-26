import os
from typing import List
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document

def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    """
    Loads PDFs using Unstructured to better handle bilingual text and tables.
    """
    documents = []
    
    if not os.path.exists(directory_path):
        print(f"❌ Error: Directory not found at {directory_path}")
        return []
    
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF(s) in {directory_path}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        try:
            print(f" Processing {pdf_file} with Unstructured...")
            
            # Using 'elements' mode to preserve table structures
            loader = UnstructuredPDFLoader(
                pdf_path,
                mode="elements", 
                strategy="hi_res", # High resolution for table detection
                infer_table_structure=True,
                languages=["ara", "fra"] # Arabic and French support
            )
            
            extracted_elements = loader.load()
            
            # Add metadata to each element
            for doc in extracted_elements:
                doc.metadata["source_file"] = pdf_file
                
                # If a table was found, ensure we use its HTML representation
                if doc.metadata.get("category") == "Table":
                    html_content = doc.metadata.get("text_as_html")
                    if html_content:
                        doc.page_content = html_content
                
                documents.append(doc)
            
            print(f"✅ Successfully extracted {len(extracted_elements)} elements from {pdf_file}")
            
        except Exception as e:
            print(f"❌ Failed to process {pdf_file}: {str(e)}")
            
    print(f"Total documents extracted: {len(documents)}")
    return documents