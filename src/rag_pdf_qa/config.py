"""Configurazione letta da variabili d'ambiente (e, se presente, dal file .env).

Nessuna credenziale è scritta nel codice: la chiave API per un LLM compatibile
OpenAI si legge solo da ``RAG_OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

try:  # python-dotenv è facoltativo
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dipende dall'ambiente
    load_dotenv = None


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Parametri della pipeline. I valori predefiniti funzionano con Ollama in locale."""

    index_dir: Path = Path("data/index")
    embedder: str = "sentence-transformers"  # oppure "tfidf"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    store: str = "faiss"  # oppure "numpy"
    llm: str = "ollama"  # oppure "openai" o "echo"
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    chunk_size: int = 800  # caratteri per chunk
    chunk_overlap: int = 150  # caratteri condivisi tra chunk consecutivi
    top_k: int = 4
    # Sotto questa similarità il sistema risponde "non lo so". Se None si usa un valore
    # adatto all'embedder (i punteggi TF-IDF sono in media più bassi di quelli semantici).
    min_score: float | None = None
    temperature: float = 0.1
    timeout: float = 120.0

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        """Crea le impostazioni leggendo le variabili ``RAG_*``."""
        if load_dotenv is not None and env_file is not None and Path(env_file).exists():
            load_dotenv(env_file, override=False)
        d = cls()
        return cls(
            index_dir=Path(_env("RAG_INDEX_DIR", str(d.index_dir))),
            embedder=_env("RAG_EMBEDDER", d.embedder),
            embedding_model=_env("RAG_EMBEDDING_MODEL", d.embedding_model),
            store=_env("RAG_STORE", d.store),
            llm=_env("RAG_LLM", d.llm),
            ollama_model=_env("RAG_OLLAMA_MODEL", d.ollama_model),
            ollama_url=_env("RAG_OLLAMA_URL", d.ollama_url),
            openai_base_url=_env("RAG_OPENAI_BASE_URL", d.openai_base_url),
            openai_model=_env("RAG_OPENAI_MODEL", d.openai_model),
            openai_api_key=os.getenv("RAG_OPENAI_API_KEY") or None,
            chunk_size=int(_env("RAG_CHUNK_SIZE", str(d.chunk_size))),
            chunk_overlap=int(_env("RAG_CHUNK_OVERLAP", str(d.chunk_overlap))),
            top_k=int(_env("RAG_TOP_K", str(d.top_k))),
            min_score=float(os.environ["RAG_MIN_SCORE"]) if os.getenv("RAG_MIN_SCORE") else None,
            temperature=float(_env("RAG_TEMPERATURE", str(d.temperature))),
            timeout=float(_env("RAG_TIMEOUT", str(d.timeout))),
        )

    def with_overrides(self, **changes: object) -> Settings:
        """Restituisce una copia con alcuni valori cambiati (ignora i ``None``)."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})
