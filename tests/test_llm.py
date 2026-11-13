import io
import json

import pytest

from rag_pdf_qa.config import Settings
from rag_pdf_qa.llm import EchoLLM, LLMError, OllamaLLM, OpenAICompatibleLLM, create_llm
from rag_pdf_qa.models import Chunk, SearchResult
from rag_pdf_qa.prompt import NOT_FOUND, build_prompt


def results() -> list[SearchResult]:
    chunk = Chunk(id="b.pdf:p2:c1", source="b.pdf", page=2, text="Il prestito dura 30 giorni. Il wifi è gratuito.")
    return [SearchResult(chunk=chunk, score=0.8)]


def test_prompt_contains_numbered_context_and_question() -> None:
    prompt = build_prompt("Quanto dura il prestito?", results())
    assert "[1] Fonte: b.pdf p.2" in prompt.user
    assert "DOMANDA: Quanto dura il prestito?" in prompt.user
    assert "SOLO" in prompt.system


def test_echo_llm_extracts_sentence_with_citation() -> None:
    answer = EchoLLM().generate(build_prompt("Quanto dura il prestito?", results()))
    assert answer.startswith("Il prestito dura 30 giorni.")
    assert "[b.pdf p.2]" in answer


def test_echo_llm_says_not_found() -> None:
    assert EchoLLM().generate(build_prompt("Chi ha vinto?", results())) == NOT_FOUND


class FakeResponse(io.BytesIO):
    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        self.close()


def test_ollama_parses_chat_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return FakeResponse(json.dumps({"message": {"content": " 30 giorni [b.pdf p.2] "}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OllamaLLM("llama3.2", "http://localhost:11434/")
    assert llm.generate(build_prompt("Quanto dura?", results())) == "30 giorni [b.pdf p.2]"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == "llama3.2" and captured["body"]["stream"] is False


def test_ollama_unreachable_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(LLMError, match="ollama serve"):
        OllamaLLM("llama3.2", "http://localhost:11434").generate(build_prompt("x", results()))


def test_openai_requires_key_and_sends_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LLMError, match="RAG_OPENAI_API_KEY"):
        OpenAICompatibleLLM("gpt-4o-mini", api_key=None)

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        assert request.headers["Authorization"] == "Bearer test-key"
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatibleLLM("gpt-4o-mini", api_key="test-key")
    assert llm.generate(build_prompt("x", results())) == "ok"


def test_factory() -> None:
    assert isinstance(create_llm(Settings(llm="echo")), EchoLLM)
    with pytest.raises(LLMError):
        create_llm(Settings(llm="gpt-magic"))
