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


def test_headings_become_separate_paragraphs() -> None:
    from rag_pdf_qa.loader import clean_text

    raw = (
        "Listino prezzi 2026\n"
        "Transfer dall'aeroporto di Bari\n"
        "Il transfer per Monopoli costa 85 euro. Il transfer per Lecce\n"
        "costa 120 euro.\n"
        "Bagagli\n"
        "Ogni passeggero porta una valigia."
    )
    text = clean_text(raw)
    paragraphs = text.split("\n\n")
    assert "Transfer dall'aeroporto di Bari" in paragraphs
    assert "Bagagli" in paragraphs
    # una riga andata a capo a metà frase resta nel suo paragrafo
    assert any("Lecce\ncosta 120 euro." in p for p in paragraphs)


def test_wrapped_line_ending_with_capitalized_word_is_not_a_heading() -> None:
    from rag_pdf_qa.loader import clean_text

    raw = "Il servizio è attivo tutto l'anno e copre le province di Bari e di\nBrindisi con auto e van."
    assert "\n\n" not in clean_text(raw)
