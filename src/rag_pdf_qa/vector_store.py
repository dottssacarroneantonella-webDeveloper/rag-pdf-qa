"""Database vettoriale: memorizza gli embedding dei chunk e trova i più simili a una domanda.

* ``FaissStore``: indice FAISS ``IndexFlatIP`` (prodotto scalare esatto). Con vettori
  normalizzati il prodotto scalare è la similarità coseno.
* ``NumpyStore``: stessa logica in NumPy puro. Va benissimo fino a qualche decina di
  migliaia di chunk e non richiede dipendenze extra.

Entrambi salvano su disco i vettori (o l'indice) e i chunk in ``chunks.json``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from rag_pdf_qa.models import Chunk, SearchResult

CHUNKS_FILE = "chunks.json"
STORE_FILE = "store.json"


class VectorStoreError(Exception):
    """Errore del database vettoriale."""


class VectorStore(ABC):
    """Interfaccia comune ai database vettoriali."""

    name: str

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Aggiunge chunk e relativi vettori (stesso ordine, vettori normalizzati)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(chunks) != vectors.shape[0]:
            raise VectorStoreError("Il numero di chunk e di vettori non coincide.")
        self._add_vectors(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        """Restituisce i ``top_k`` chunk più simili, dal più al meno rilevante."""
        if not self.chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(top_k, len(self.chunks))
        scores, indices = self._search(query, k)
        return [
            SearchResult(chunk=self.chunks[i], score=float(s))
            for s, i in zip(scores, indices, strict=True)
            if i >= 0
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in self.chunks]
        (directory / CHUNKS_FILE).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (directory / STORE_FILE).write_text(json.dumps({"name": self.name}), encoding="utf-8")
        self._save_vectors(directory)

    def _load_chunks(self, directory: Path) -> None:
        data = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(d) for d in data]

    @abstractmethod
    def _add_vectors(self, vectors: np.ndarray) -> None: ...

    @abstractmethod
    def _search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def _save_vectors(self, directory: Path) -> None: ...


class NumpyStore(VectorStore):
    """Ricerca esatta per similarità coseno con NumPy."""

    name = "numpy"
    _VECTORS_FILE = "vectors.npy"

    def __init__(self) -> None:
        super().__init__()
        self._vectors: np.ndarray | None = None

    def _add_vectors(self, vectors: np.ndarray) -> None:
        if self._vectors is not None and self._vectors.shape[1] != vectors.shape[1]:
            raise VectorStoreError("Dimensione dei vettori diversa da quella dell'indice.")
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def _search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._vectors is not None
        scores = self._vectors @ query[0]
        top = np.argsort(-scores)[:k]
        return scores[top], top

    def _save_vectors(self, directory: Path) -> None:
        if self._vectors is not None:
            np.save(directory / self._VECTORS_FILE, self._vectors)

    @classmethod
    def load(cls, directory: Path) -> NumpyStore:
        store = cls()
        store._load_chunks(directory)
        store._vectors = np.load(directory / cls._VECTORS_FILE)
        return store


class FaissStore(VectorStore):
    """Indice FAISS a prodotto scalare (import solo quando serve)."""

    name = "faiss"
    _INDEX_FILE = "index.faiss"

    def __init__(self) -> None:
        super().__init__()
        self._faiss = self._import_faiss()
        self._index = None

    @staticmethod
    def _import_faiss():  # type: ignore[no-untyped-def]
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError(
                "FAISS non è installato. Esegui: pip install faiss-cpu oppure usa --store numpy"
            ) from exc
        return faiss

    def _add_vectors(self, vectors: np.ndarray) -> None:
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(vectors.shape[1])
        elif self._index.d != vectors.shape[1]:
            raise VectorStoreError("Dimensione dei vettori diversa da quella dell'indice.")
        self._index.add(np.ascontiguousarray(vectors))

    def _search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, indices = self._index.search(np.ascontiguousarray(query), k)
        return scores[0], indices[0]

    def _save_vectors(self, directory: Path) -> None:
        if self._index is not None:
            self._faiss.write_index(self._index, str(directory / self._INDEX_FILE))

    @classmethod
    def load(cls, directory: Path) -> FaissStore:
        store = cls()
        store._load_chunks(directory)
        store._index = store._faiss.read_index(str(directory / cls._INDEX_FILE))
        return store


def create_store(kind: str) -> VectorStore:
    """Factory: crea il database vettoriale richiesto."""
    if kind == "faiss":
        return FaissStore()
    if kind == "numpy":
        return NumpyStore()
    raise VectorStoreError(f"Store sconosciuto: {kind!r} (usa 'faiss' o 'numpy')")


def load_store(directory: Path) -> VectorStore:
    """Ricarica il database vettoriale salvato con ``save``."""
    meta = directory / STORE_FILE
    if not meta.exists():
        raise VectorStoreError(f"Indice non trovato in {directory}: esegui prima 'ingest'.")
    name = json.loads(meta.read_text(encoding="utf-8"))["name"]
    if name == "faiss":
        return FaissStore.load(directory)
    if name == "numpy":
        return NumpyStore.load(directory)
    raise VectorStoreError(f"Formato di indice sconosciuto: {name!r}")
