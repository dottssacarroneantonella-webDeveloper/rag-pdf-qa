from pathlib import Path

import pytest

from rag_pdf_qa.config import Settings


def test_defaults_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_LLM", "echo")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_OPENAI_API_KEY", "")
    settings = Settings.from_env(env_file=tmp_path / "non-esiste.env")
    assert settings.llm == "echo" and settings.top_k == 7
    assert settings.openai_api_key is None
    assert settings.embedder == "sentence-transformers"


def test_overrides_ignore_none() -> None:
    settings = Settings().with_overrides(llm="echo", top_k=None)
    assert settings.llm == "echo" and settings.top_k == Settings().top_k
