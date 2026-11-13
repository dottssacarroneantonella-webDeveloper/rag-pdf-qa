from pathlib import Path

import pytest

from rag_pdf_qa.loader import LoaderError, clean_text, load_directory, load_pdf


def test_load_pdf_returns_pages_with_metadata(example_pdf: Path) -> None:
    pages = load_pdf(example_pdf)
    assert [p.page for p in pages] == [1, 2, 3]
    assert all(p.source == "regolamento_biblioteca.pdf" for p in pages)
    assert "Orari di apertura" in pages[0].text


def test_load_directory(example_pdf: Path) -> None:
    assert len(load_directory(example_pdf.parent)) >= 3


def test_clean_text_joins_hyphenated_words() -> None:
    assert clean_text("infor-\nmatica   e  dati") == "informatica e dati"


def test_missing_file_and_folder(tmp_path: Path) -> None:
    with pytest.raises(LoaderError):
        load_pdf(tmp_path / "manca.pdf")
    with pytest.raises(LoaderError):
        load_directory(tmp_path / "manca")
    with pytest.raises(LoaderError, match="Nessun PDF"):
        load_directory(tmp_path)


def test_invalid_pdf(tmp_path: Path) -> None:
    fake = tmp_path / "rotto.pdf"
    fake.write_text("non sono un pdf")
    with pytest.raises(LoaderError):
        load_pdf(fake)
