import os
import threading
from typing import Any, Dict, List
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import configure_console, initialize_rag, load_config
from src.rag_pipeline import answer_question
from src.text_splitter import normalize_arabic_text


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceRow(BaseModel):
    source: str = ""
    row: Any = ""
    code: str = ""
    program: str = ""
    university: str = ""
    establishment: str = ""
    type: str = ""


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceRow]
    timings: Dict[str, float] = {}


class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool


app = FastAPI(title="Orientation RAG API", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag_pipeline = None
_rag_lock = threading.Lock()


def _get_rag_pipeline():
    global _rag_pipeline

    if _rag_pipeline is None:
        with _rag_lock:
            if _rag_pipeline is None:
                configure_console()
                config = load_config()
                _rag_pipeline, _ = initialize_rag(config, force_reindex=False)

    return _rag_pipeline


def _format_sources(source_documents) -> List[Dict[str, Any]]:
    rows = []

    for doc in source_documents:
        metadata = doc.metadata
        rows.append(
            {
                "source": metadata.get("source_file", ""),
                "row": metadata.get("row_id", metadata.get("row_index", "")),
                "code": metadata.get("program_code", ""),
                "program": metadata.get("program_ar") or metadata.get("program_fr", ""),
                "university": metadata.get("university", ""),
                "establishment": metadata.get("faculty", ""),
                "type": metadata.get("doc_type", ""),
            }
        )

    return rows


@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(status="ok", rag_loaded=_rag_pipeline is not None)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", rag_loaded=_rag_pipeline is not None)


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    question = normalize_arabic_text(request.question.strip())

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        rag_pipeline = _get_rag_pipeline()
        result = await asyncio.to_thread(answer_question, rag_pipeline, question)
    except SystemExit as exc:
        raise HTTPException(status_code=500, detail="RAG initialization failed.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnswerResponse(
        answer=result["answer"],
        sources=_format_sources(result.get("source_documents", [])),
        timings=result.get("timings", {}),
    )
