# RAG with Gemma - CSV Question Answering

LangChain RAG over CSV orientation-guide data. Rows are embedded into ChromaDB, then answered with a Google/Gemma chat model.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
CSV_DIRECTORY=data
CHROMA_DB_DIR=chroma_db
EMBEDDING_MODEL=gemini-embedding-001
GEMMA_MODEL=gemma-4-26b-a4b-it
CHUNK_SIZE=3000
CHUNK_OVERLAP=500
CSV_DOCUMENT_MODE=summary
EMBEDDING_BATCH_SIZE=50
EMBEDDING_REQUESTS_PER_MINUTE=60
```

```bash
python main.py --reindex
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```
