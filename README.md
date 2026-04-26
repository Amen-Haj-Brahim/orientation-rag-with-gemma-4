# RAG with Gemma - PDF Question Answering

A Retrieval-Augmented Generation (RAG) system using Google's Gemma model to query PDF documents. Supports Arabic text.

## Features

- **Gemma 4** via Google AI Studio
- **LangChain** for RAG pipeline
- **ChromaDB** for vector storage
- **Arabic text support** with normalization
- PDF loading with source tracking

## Prerequisites

1. Python 3.10+
2. Google AI Studio API key

## Installation

### 1. Clone/Create Project

```bash
mkdir gemma-rag
cd gemma-rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

```bash
copy .env.example .env
```

Edit `.env` and set your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from: https://aistudio.google.com/app/apikey

### 5. Add PDF Files

Place your PDF files in the `data/` directory:

```
project/
├── data/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
```

## Usage

### Run the Application

```bash
python main.py
```

### Options

- `--reindex` - Force re-indexing of all PDFs (useful when adding new documents)

```bash
python main.py --reindex
```

### Example Session

```
============================================================
RAG with Gemma - PDF Question Answering
============================================================
Loading PDFs from: data
Loaded: document.pdf (10 pages)
Split 10 documents into 45 chunks
Created vector store with 45 documents
Initializing Gemma model...
Creating RAG pipeline...

============================================================
Ready! Enter your questions (or 'quit' to exit)
============================================================

You: What is this document about?

Thinking...

Answer:
This document discusses the main topics covered in the PDF files...
[Content from retrieved documents]

Sources:
- document1.pdf
- document2.pdf

You: quit
Goodbye!
```

## Project Structure

```
project/
├── .env.example          # Environment template
├── requirements.txt     # Python dependencies
├── main.py             # Entry point
├── README.md           # This file
├── data/               # PDF input directory
├── chroma_db/          # Vector database (created on first run)
└── src/
    ├── pdf_loader.py       # PDF loading
    ├── text_splitter.py    # Text chunking
    ├── vector_store.py    # ChromaDB setup
    ├── retrieval.py       # Document retrieval
    └── rag_pipeline.py    # RAG chain
```

## Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Your Google API key |
| `PDF_DIRECTORY` | `data` | Directory containing PDFs |
| `CHROMA_DB_DIR` | `chroma_db` | Vector database directory |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `GEMMA_MODEL` | `gemma-4-2b` | Gemma model |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |

## Arabic Support

The system includes:
- Arabic text normalization (removes diacritics)
- Arabic-aware text splitting
- Bilingual prompt templates (Arabic/English)

## Troubleshooting

### No API Key

```
Error: Please set your GOOGLE_API_KEY in the .env file
```

**Solution**: Add your API key to `.env`

### No PDF Files Found

```
Error: No PDF files found in: data
```

**Solution**: Create `data/` directory and add PDF files

### Vector Store Error

If you encounter vector store errors, try re-indexing:

```bash
python main.py --reindex
```

## License

MIT"# orientation-rag-with-gemma-4" 
