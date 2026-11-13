import pytest

from rag_pdf_qa.chunker import chunk_page, chunk_pages
from rag_pdf_qa.models import Page


def make_page(text: str, page: int = 1) -> Page:
    return Page(source="doc.pdf", page=page, text=text)


def test_short_text_is_single_chunk() -> None:
    chunks = chunk_page(make_page("Una frase breve."), chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].id == "doc.pdf:p1:c1"
    assert chunks[0].citation == "doc.pdf p.1"


def test_chunks_respect_max_size_and_overlap() -> None:
    sentences = [f"Questa è la frase numero {i} del documento di prova." for i in range(40)]
    chunks = chunk_page(make_page(" ".join(sentences)), chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    # la coda di ogni chunk ricompare all'inizio del successivo (overlap)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text[-20:].split()[-1] in current.text


def test_no_text_is_lost() -> None:
    words = [f"parola{i}" for i in range(300)]
    chunks = chunk_page(make_page(" ".join(words)), chunk_size=120, overlap=0)
    joined = " ".join(c.text for c in chunks)
    assert all(w in joined for w in words)


def test_metadata_is_kept_across_pages() -> None:
    chunks = chunk_pages([make_page("Pagina uno.", 1), make_page("Pagina due.", 2)])
    assert [c.page for c in chunks] == [1, 2]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_page(make_page("testo"), chunk_size=size, overlap=overlap)
