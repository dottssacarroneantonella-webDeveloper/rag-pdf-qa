from pathlib import Path

import pytest

from rag_pdf_qa.cli import main


def test_cli_ingest_and_ask(tmp_path: Path, example_pdf: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index = str(tmp_path / "idx")
    assert main(["--index-dir", index, "ingest", str(example_pdf.parent), "--embedder", "tfidf", "--store", "numpy"]) == 0
    assert "Indicizzati 1 documenti" in capsys.readouterr().out

    assert main(["--index-dir", index, "ask", "Quanto dura il prestito dei libri?", "--llm", "echo", "--show-context"]) == 0
    out = capsys.readouterr().out
    assert "30 giorni" in out and "Fonti:" in out and "Chunk recuperati" in out


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--index-dir", str(tmp_path / "vuoto"), "ask", "ciao", "--llm", "echo"]) == 1
    assert "Errore" in capsys.readouterr().err
