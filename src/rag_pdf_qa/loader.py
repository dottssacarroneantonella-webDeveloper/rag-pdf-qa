"""Estrazione del testo dai PDF, pagina per pagina."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_pdf_qa.models import Page

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Errore nella lettura dei documenti."""


def clean_text(text: str) -> str:
    """Normalizza il testo estratto: unisce le parole spezzate a fine riga e gli spazi multipli."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "infor-\nmatica" -> "informatica"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: str | Path) -> list[Page]:
    """Legge un PDF e restituisce una ``Page`` per ogni pagina che contiene testo.

    Le pagine senza testo (per esempio scansioni senza OCR) vengono saltate con un avviso.
    """
    path = Path(path)
    if not path.is_file():
        raise LoaderError(f"File non trovato: {path}")
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError) as exc:
        raise LoaderError(f"Impossibile leggere {path.name}: {exc}") from exc
    if reader.is_encrypted:
        raise LoaderError(f"{path.name} è protetto da password: rimuovi la protezione e riprova.")

    pages: list[Page] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = clean_text(pdf_page.extract_text() or "")
        if text:
            pages.append(Page(source=path.name, page=number, text=text))
        else:
            logger.warning("%s p.%d: nessun testo estraibile (serve OCR?)", path.name, number)
    return pages


def load_directory(directory: str | Path) -> list[Page]:
    """Legge tutti i PDF di una cartella (sottocartelle comprese), in ordine alfabetico."""
    directory = Path(directory)
    if not directory.is_dir():
        raise LoaderError(f"Cartella non trovata: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() == ".pdf")
    if not files:
        raise LoaderError(f"Nessun PDF trovato in {directory}")
    pages: list[Page] = []
    for file in files:
        pages.extend(load_pdf(file))
    if not pages:
        raise LoaderError("I PDF non contengono testo estraibile.")
    return pages
