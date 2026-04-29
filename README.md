# RAG with Gemma - CSV Question Answering

This project builds a LangChain RAG pipeline over CSV orientation-guide data.
It loads rows from `data/*.csv`, converts each row into semantic documents,
embeds them with Google embeddings, stores them in ChromaDB, and answers
questions with a Google/Gemma chat model.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
CSV_DIRECTORY=data
CHROMA_DB_DIR=chroma_db
EMBEDDING_MODEL=gemini-embedding-001
GEMMA_MODEL=gemma-4-26b-a4b-it
CHUNK_SIZE=3000
CHUNK_OVERLAP=500
```

3. Put CSV files in `data/`.

4. Rebuild the vector store after changing CSV loading or text normalization:

```bash
python main.py --reindex
```

5. Run normally:

```bash
python main.py
```

## Debug Retrieval

Run:

```bash
python test.py
```

The script rebuilds `chroma_db`, prints sample loaded documents, and tests a few
Arabic/French retrieval queries.

## Notes

- The CSV should be UTF-8. The included loader uses `utf-8-sig` to handle files
  with a BOM.
- If Arabic appears as `Ø§Ù...` in old Windows PowerShell output, check the
  actual Python value with `unicode_escape`; the data can still be correct
  internally while the terminal display is wrong.
- The RAG chain returns source documents, including CSV file, row id, program
  code, university, establishment, and domain metadata where available.
