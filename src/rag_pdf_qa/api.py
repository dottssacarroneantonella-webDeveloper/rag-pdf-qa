"""API REST facoltativa con FastAPI.

Avvio::

    pip install "fastapi" "uvicorn[standard]"
    uvicorn rag_pdf_qa.api:app --reload

Documentazione interattiva su http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pdf_qa.config import Settings
from rag_pdf_qa.embeddings import EmbedderError
from rag_pdf_qa.llm import LLMError
from rag_pdf_qa.loader import LoaderError
from rag_pdf_qa.pipeline import RagPipeline
from rag_pdf_qa.vector_store import VectorStoreError

app = FastAPI(title="rag-pdf-qa", description="Domande e risposte sui PDF con RAG", version="0.1.0")
_pipeline: RagPipeline | None = None

# Solo le cartelle sotto questa radice possono essere indicizzate via API.
DOCS_ROOT = Path("data/pdf").resolve()


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(Settings.from_env())
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    citation: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    answered: bool
    sources: list[Source]


class IngestRequest(BaseModel):
    path: str = Field(default=".", description="sottocartella di data/pdf da indicizzare")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        answer = get_pipeline().ask(request.question, request.top_k)
    except (EmbedderError, VectorStoreError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AskResponse(
        answer=answer.text,
        answered=answer.answered,
        sources=[Source(citation=r.chunk.citation, score=round(r.score, 4), text=r.chunk.text) for r in answer.sources],
    )


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    target = (DOCS_ROOT / request.path).resolve()
    if not target.is_relative_to(DOCS_ROOT):  # blocca percorsi tipo "../../"
        raise HTTPException(status_code=400, detail="Percorso non consentito.")
    try:
        report = get_pipeline().ingest(target)
    except LoaderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EmbedderError, VectorStoreError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"documents": report.documents, "pages": report.pages, "chunks": report.chunks}
