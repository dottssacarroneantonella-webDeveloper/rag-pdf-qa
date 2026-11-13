"""Strutture dati condivise da tutti i moduli della pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Page:
    """Testo di una singola pagina di un PDF."""

    source: str  # nome del file, es. "manuale.pdf"
    page: int  # numero di pagina, a partire da 1
    text: str


@dataclass(frozen=True)
class Chunk:
    """Porzione di testo indicizzata, con i metadati necessari a citare la fonte."""

    id: str  # identificativo stabile, es. "manuale.pdf:p3:c2"
    source: str
    page: int
    text: str

    @property
    def citation(self) -> str:
        """Etichetta della fonte da mostrare nella risposta, es. "manuale.pdf p.3"."""
        return f"{self.source} p.{self.page}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(id=data["id"], source=data["source"], page=int(data["page"]), text=data["text"])


@dataclass(frozen=True)
class SearchResult:
    """Un chunk recuperato con il suo punteggio di similarità coseno (da -1 a 1)."""

    chunk: Chunk
    score: float


@dataclass
class Answer:
    """Risposta finale restituita all'utente."""

    question: str
    text: str
    sources: list[SearchResult] = field(default_factory=list)
    answered: bool = True  # False quando il sistema risponde "non lo so"

    def citations(self) -> list[str]:
        """Elenco delle fonti senza duplicati, nell'ordine di rilevanza."""
        seen: list[str] = []
        for result in self.sources:
            if result.chunk.citation not in seen:
                seen.append(result.chunk.citation)
        return seen
