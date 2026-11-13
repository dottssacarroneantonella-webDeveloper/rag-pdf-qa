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


_HEADING_MAX_CHARS = 80
_HEADING_MAX_WORDS = 10
_ENDS_SENTENCE = re.compile(r"[.!?:;]$")
_ENDS_LINE_WITH_PUNCT = re.compile(r"[.!?:;,]$")
# una riga che finisce con un articolo, una preposizione o una congiunzione continua sotto
_CONTINUES = {
    "a", "ad", "al", "alla", "alle", "agli", "ai", "che", "con", "da", "dal", "dalla", "dalle", "dei",
    "del", "della", "delle", "degli", "di", "e", "ed", "gli", "i", "il", "in", "la", "le", "lo", "nel",
    "nella", "nelle", "o", "per", "su", "sul", "sulla", "tra", "fra", "un", "una", "uno",
    "and", "of", "the", "to", "for", "with", "or",
}


def _is_heading(line: str, previous_ends_block: bool, following: str | None) -> bool:
    """Una riga è un titolo se è corta, non termina con punteggiatura né con una parola che
    richiede un seguito, la precede la fine di una frase o un altro titolo (o l'inizio della
    pagina) e la riga dopo comincia con maiuscola o cifra.

    Le righe di testo andate a capo invece continuano con una minuscola, sono lunghe quanto
    la colonna o finiscono con "di", "e", "per"…, quindi restano unite al paragrafo.
    """
    words = line.split()
    if not words or len(line) > _HEADING_MAX_CHARS or len(words) > _HEADING_MAX_WORDS:
        return False
    if _ENDS_LINE_WITH_PUNCT.search(line) or words[-1].lower() in _CONTINUES:
        return False
    if not previous_ends_block:
        return False
    return following is None or following[:1].isupper() or following[:1].isdigit()


def separate_headings(text: str) -> str:
    """Mette i titoli in paragrafi separati (riga vuota prima e dopo), così non vengono
    incollati alla frase successiva: "Bagagli\nOgni passeggero..." → "Bagagli\n\nOgni passeggero..."."""
    lines = [line.strip() for line in text.split("\n")]
    out: list[str] = []
    previous_ends_block = True  # inizio pagina
    for i, line in enumerate(lines):
        if not line:
            out.append("")
            previous_ends_block = True
            continue
        following = next((nxt for nxt in lines[i + 1 :] if nxt), None)
        if _is_heading(line, previous_ends_block, following):
            out.extend(["", line, ""])
            previous_ends_block = True
        else:
            out.append(line)
            previous_ends_block = bool(_ENDS_SENTENCE.search(line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def clean_text(text: str) -> str:
    """Normalizza il testo estratto: parole spezzate a fine riga, spazi multipli e titoli."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "infor-\nmatica" -> "informatica"
    text = re.sub(r"[ \t]+", " ", text)
    text = separate_headings(text)
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
