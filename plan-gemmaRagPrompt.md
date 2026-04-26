# Plan: RAG with Gemma and PDF Files

**Goal**: Build a RAG system using Google's Gemma model via Google AI Studio, LangChain for pipeline, ChromaDB for vector storage, to query PDF documents.

**User Requirements**:

- PDF directory: `F:\9rayox\gemma rag\data`
- Model: Gemma via Google AI Studio
- Framework: LangChain
- Vector store: ChromaDB
- **PDF content**: Arabic text

## Steps

### Phase 1: Project Setup

1. Create `requirements.txt` with dependencies (langchain-google-genai, chromadb, pypdf, python-dotenv)
2. Create `.env` template for API key storage
3. Initialize project structure

### Phase 2: PDF Processing

4. Create `pdf_loader.py` — load PDFs from `data/` directory using PyPDFLoader
5. Create `text_splitter.py` — split documents into chunks using RecursiveCharacterTextSplitter

### Phase 3: Vector Store

6. Create `vector_store.py` — embed chunks and store in ChromaDB
7. Create `retrieval.py` — retrieve relevant chunks based on query

### Phase 4: RAG Pipeline

8. Create `rag_pipeline.py` — combine retrieval + Gemma for Q&A
9. Create `main.py` — CLI interface for querying PDFs

### Phase 5: Documentation

10. Create `README.md` — setup and usage instructions

## Relevant Files

- `requirements.txt` — Python dependencies
- `.env` — API key configuration
- `src/pdf_loader.py` — PDF loading logic
- `src/text_splitter.py` — chunking logic
- `src/vector_store.py` — ChromaDB setup
- `src/retrieval.py` — retrieval logic
- `src/rag_pipeline.py` — RAG chain
- `main.py` — entry point
- `data/` — PDF input directory (user provides)

## Verification

1. Run `pip install -r requirements.txt`
2. Add Google AI Studio API key to `.env`
3. Place PDF files in `data/` folder
4. Run `python main.py` — should prompt for query
5. Ask a question about your PDF content — verify relevant answer

## Decisions

- Using `langchain-google-genai` for Google AI Studio integration
- ChromaDB persists to `chroma_db/` directory
- Default chunk size: 1000, overlap: 200
- Gemma **4** model (latest, supports Arabic)

## Further Considerations

1. If you need embeddings beyond Google, can switch to OpenAI or Hugging Face embeddings
2. For large PDF volumes, consider batch processing
3. **Arabic support**: Gemma 4 has improved multilingual capabilities; may need to verify Arabic tokenization
4. Consider adding Arabic-specific text preprocessing (normalization, diacritics handling)
5. For Arabic PDFs, ensure font encoding is handled correctly during extraction
