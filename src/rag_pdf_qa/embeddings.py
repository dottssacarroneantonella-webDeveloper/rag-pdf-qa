"""Calcolo degli embedding: trasforma un testo in un vettore numerico.

Due implementazioni con la stessa interfaccia:

* ``SentenceTransformerEmbedder``: embedding semantici con un modello multilingue
  (adatto a PDF in italiano). È la scelta per l'uso reale.
* ``TfidfEmbedder``: vettori TF-IDF di scikit-learn. Non capisce i sinonimi, ma non
  scarica modelli e funziona offline: serve per i test e per una demo immediata.

Tutti i vettori restituiti sono normalizzati (norma L2 = 1), quindi il prodotto
scalare tra due vettori coincide con la similarità coseno.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from rag_pdf_qa.text_utils import ITALIAN_STOPWORDS

EMBEDDER_FILE = "embedder.json"


class EmbedderError(Exception):
    """Errore nella creazione o nell'uso di un embedder."""


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalizza ogni riga a norma 1 (le righe nulle restano nulle)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class Embedder(ABC):
    """Interfaccia comune a tutti gli embedder."""

    name: str

    def fit(self, texts: list[str]) -> None:  # noqa: B027 - facoltativo per le sottoclassi
        """Adatta l'embedder al corpus (serve solo a TF-IDF)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Restituisce una matrice ``len(texts) x dimensione`` di vettori normalizzati."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embedding di una singola domanda, come vettore 1-D."""
        return self.embed([text])[0]

    def save(self, directory: Path) -> None:
        """Salva ciò che serve per ricaricare lo stesso embedder."""
        (directory / EMBEDDER_FILE).write_text(json.dumps(self._config()), encoding="utf-8")

    @abstractmethod
    def _config(self) -> dict[str, str]: ...


class TfidfEmbedder(Embedder):
    """Embedding lessicali TF-IDF (unigrammi e bigrammi)."""

    name = "tfidf"
    _VECTORIZER_FILE = "tfidf.pkl"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=sorted(ITALIAN_STOPWORDS),
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        if not texts:
            raise EmbedderError("Impossibile adattare TF-IDF su un corpus vuoto.")
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise EmbedderError("TfidfEmbedder non adattato: esegui prima l'indicizzazione.")
        matrix = self._vectorizer.transform(texts).toarray()
        return l2_normalize(matrix)

    def save(self, directory: Path) -> None:
        super().save(directory)
        with open(directory / self._VECTORIZER_FILE, "wb") as fh:
            pickle.dump(self._vectorizer, fh)

    @classmethod
    def load(cls, directory: Path) -> TfidfEmbedder:
        """Ricarica il vettorizzatore salvato. Carica solo indici creati da te (pickle)."""
        embedder = cls()
        with open(directory / cls._VECTORIZER_FILE, "rb") as fh:
            embedder._vectorizer = pickle.load(fh)  # noqa: S301 - file generato da questo progetto
        embedder._fitted = True
        return embedder

    def _config(self) -> dict[str, str]:
        return {"name": self.name}


class SentenceTransformerEmbedder(Embedder):
    """Embedding semantici con sentence-transformers (import solo quando serve)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderError(
                "sentence-transformers non è installato. Esegui: pip install sentence-transformers "
                "oppure usa --embedder tfidf"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def _config(self) -> dict[str, str]:
        return {"name": self.name, "model": self.model_name}


def create_embedder(kind: str, model_name: str | None = None) -> Embedder:
    """Factory: crea l'embedder richiesto dalla configurazione."""
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name or "paraphrase-multilingual-MiniLM-L12-v2")
    raise EmbedderError(f"Embedder sconosciuto: {kind!r} (usa 'sentence-transformers' o 'tfidf')")


def load_embedder(directory: Path) -> Embedder:
    """Ricrea l'embedder usato in fase di indicizzazione, così domanda e documenti
    finiscono nello stesso spazio vettoriale."""
    config_path = directory / EMBEDDER_FILE
    if not config_path.exists():
        raise EmbedderError(f"Indice non trovato in {directory}: esegui prima 'ingest'.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["name"] == "tfidf":
        return TfidfEmbedder.load(directory)
    return create_embedder(config["name"], config.get("model"))
