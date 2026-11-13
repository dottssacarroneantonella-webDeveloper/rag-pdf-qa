import json
from pathlib import Path

import pytest

from rag_pdf_qa.config import Settings
from rag_pdf_qa.pipeline import RagPipeline


def test_ingest_writes_index(offline_settings: Settings, example_pdf: Path) -> None:
    report = RagPipeline(offline_settings).ingest(example_pdf)
    assert report.documents == 1 and report.pages == 3 and report.chunks >= 3
    manifest = json.loads((offline_settings.index_dir / "manifest.json").read_text())
    assert manifest["documents"] == ["regolamento_biblioteca.pdf"]
    assert manifest["embedder"] == "tfidf"


@pytest.mark.parametrize(
    "question,expected_text,expected_page",
    [
        ("Quali sono gli orari di apertura della biblioteca?", "9:00", 1),
        ("Quanti giorni dura il prestito dei libri?", "30 giorni", 2),
        ("Quanti posti ha la sala studio?", "40 posti", 3),
    ],
)
def test_answers_cite_the_right_page(
    indexed_pipeline: RagPipeline, question: str, expected_text: str, expected_page: int
) -> None:
    answer = indexed_pipeline.ask(question)
    assert answer.answered
    assert expected_text in answer.text
    assert answer.sources[0].chunk.page == expected_page
    assert f"regolamento_biblioteca.pdf p.{expected_page}" in answer.text


def test_out_of_scope_question_returns_not_found(indexed_pipeline: RagPipeline) -> None:
    answer = indexed_pipeline.ask("Qual è la capitale della Francia?")
    assert not answer.answered
    assert answer.text.startswith("Non lo so")


def test_index_is_reloaded_from_disk(indexed_pipeline: RagPipeline) -> None:
    fresh = RagPipeline(indexed_pipeline.settings)  # nuova istanza: legge l'indice salvato
    assert "30 giorni" in fresh.ask("Quanto dura il prestito?").text


def test_empty_question(indexed_pipeline: RagPipeline) -> None:
    with pytest.raises(ValueError):
        indexed_pipeline.ask("   ")
