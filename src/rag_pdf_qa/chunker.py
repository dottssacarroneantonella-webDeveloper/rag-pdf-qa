"""Suddivisione del testo in chunk sovrapposti, rispettando paragrafi e frasi."""

from __future__ import annotations

import re

from rag_pdf_qa.models import Chunk, Page

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


def _split_units(text: str, max_len: int) -> list[str]:
    """Divide il testo in unità (paragrafi, poi frasi, poi parole) non più lunghe di ``max_len``."""
    units: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        paragraph = " ".join(paragraph.split())
        if len(paragraph) <= max_len:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            if len(sentence) <= max_len:
                units.append(sentence)
            else:  # frase lunghissima: si spezza sulle parole
                words, current = sentence.split(), ""
                for word in words:
                    if current and len(current) + 1 + len(word) > max_len:
                        units.append(current)
                        current = word
                    else:
                        current = f"{current} {word}".strip()
                if current:
                    units.append(current)
    return units


def _tail(text: str, size: int) -> str:
    """Ultimi ``size`` caratteri del testo, senza tagliare una parola a metà."""
    if size <= 0 or len(text) <= size:
        return text if size > 0 else ""
    cut = text[-size:]
    space = cut.find(" ")
    return cut[space + 1 :] if space != -1 else cut


def chunk_page(page: Page, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Crea i chunk di una pagina.

    Le unità di testo vengono accumulate finché non si supera ``chunk_size``; ogni nuovo
    chunk riparte con gli ultimi ``overlap`` caratteri del precedente, così una frase
    a cavallo tra due chunk non perde il contesto.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size deve essere positivo")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap deve essere compreso tra 0 e chunk_size - 1")

    texts: list[str] = []
    current = ""
    for unit in _split_units(page.text, chunk_size):
        # i paragrafi (e i titoli) restano su righe diverse: non diventano parte della frase dopo
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        texts.append(current)
        prefix = _tail(current, overlap)
        current = f"{prefix}\n{unit}" if prefix and len(prefix) + 1 + len(unit) <= chunk_size else unit
    if current:
        texts.append(current)

    return [
        Chunk(id=f"{page.source}:p{page.page}:c{i}", source=page.source, page=page.page, text=t)
        for i, t in enumerate(texts, start=1)
    ]


def chunk_pages(pages: list[Page], chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Applica ``chunk_page`` a tutte le pagine, mantenendo l'ordine."""
    return [c for page in pages for c in chunk_page(page, chunk_size, overlap)]
