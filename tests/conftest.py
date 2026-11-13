"""Fixture comuni: PDF di esempio e pipeline offline (TF-IDF + NumPy + EchoLLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_pdf_qa.config import Settings
from rag_pdf_qa.pipeline import RagPipeline

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def example_pdf() -> Path:
    return EXAMPLES / "regolamento_biblioteca.pdf"


@pytest.fixture
def offline_settings(tmp_path: Path) -> Settings:
    """Configurazione che non scarica modelli e non usa la rete."""
    return Settings(
        index_dir=tmp_path / "index",
        embedder="tfidf",
        store="numpy",
        llm="echo",
        chunk_size=400,
        chunk_overlap=80,
        top_k=3,
    )


@pytest.fixture
def indexed_pipeline(offline_settings: Settings, example_pdf: Path) -> RagPipeline:
    pipeline = RagPipeline(offline_settings)
    pipeline.ingest(example_pdf.parent)
    return pipeline
