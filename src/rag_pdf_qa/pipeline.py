"""Orchestrazione della pipeline RAG: indicizzazione (ingest) e risposta (ask)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from rag_pdf_qa.chunker import chunk_pages
from rag_pdf_qa.config import Settings
from rag_pdf_qa.embeddings import Embedder, create_embedder, load_embedder
from rag_pdf_qa.llm import LLM, create_llm
from rag_pdf_qa.loader import load_directory, load_pdf
from rag_pdf_qa.models import Answer, Page, SearchResult
from rag_pdf_qa.prompt import NOT_FOUND, build_prompt
from rag_pdf_qa.vector_store import VectorStore, create_store, load_store

logger = logging.getLogger(__name__)
MANIFEST_FILE = "manifest.json"

# Soglie di similarità predefinite, tarate sul PDF di esempio.
DEFAULT_MIN_SCORE = {"sentence-transformers": 0.30, "tfidf": 0.10}


@dataclass(frozen=True)
class IngestReport:
    """Riepilogo dell'indicizzazione."""

    documents: int
    pages: int
    chunks: int
    seconds: float
    index_dir: Path


class RagPipeline:
    """Collega loader, chunker, embedder, database vettoriale e LLM."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.settings = settings
        self._embedder = embedder
        self._store = store
        self._llm = llm

    # ---------- indicizzazione ----------

    def ingest(self, path: str | Path) -> IngestReport:
        """Indicizza un PDF o una cartella di PDF e salva l'indice in ``settings.index_dir``.

        L'indice viene ricostruito da zero: così non restano chunk di documenti rimossi.
        """
        start = time.perf_counter()
        path = Path(path)
        pages: list[Page] = load_pdf(path) if path.is_file() else load_directory(path)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        logger.info("%d pagine -> %d chunk", len(pages), len(chunks))

        embedder = create_embedder(self.settings.embedder, self.settings.embedding_model)
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        vectors = embedder.embed(texts)

        store = create_store(self.settings.store)
        store.add(chunks, vectors)

        index_dir = self.settings.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        store.save(index_dir)
        embedder.save(index_dir)
        documents = sorted({p.source for p in pages})
        manifest = {
            "documents": documents,
            "pages": len(pages),
            "chunks": len(chunks),
            "embedder": embedder.name,
            "store": store.name,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
        }
        (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        self._embedder, self._store = embedder, store
        return IngestReport(
            documents=len(documents),
            pages=len(pages),
            chunks=len(chunks),
            seconds=time.perf_counter() - start,
            index_dir=index_dir,
        )

    # ---------- risposta ----------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = load_embedder(self.settings.index_dir)
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = load_store(self.settings.index_dir)
        return self._store

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    @property
    def min_score(self) -> float:
        """Soglia di similarità effettiva (da configurazione o predefinita per l'embedder)."""
        if self.settings.min_score is not None:
            return self.settings.min_score
        return DEFAULT_MIN_SCORE.get(self.embedder.name, 0.25)

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        """Recupera i chunk più simili alla domanda (senza chiamare l'LLM)."""
        query_vector = self.embedder.embed_query(question)
        return self.store.search(query_vector, top_k or self.settings.top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """Risponde a una domanda usando solo i documenti indicizzati."""
        question = question.strip()
        if not question:
            raise ValueError("La domanda è vuota.")
        results = self.retrieve(question, top_k)
        relevant = [r for r in results if r.score >= self.min_score]
        if not relevant:
            # Nessun chunk abbastanza simile: inutile (e rischioso) chiamare il modello.
            return Answer(question=question, text=NOT_FOUND, sources=results, answered=False)
        text = self.llm.generate(build_prompt(question, relevant))
        answered = not text.startswith("Non lo so")
        return Answer(question=question, text=text, sources=relevant, answered=answered)
