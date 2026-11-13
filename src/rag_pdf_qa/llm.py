"""Client per i modelli linguistici (LLM) che generano la risposta finale.

* ``OllamaLLM``: modello locale e gratuito tramite Ollama (https://ollama.com).
* ``OpenAICompatibleLLM``: qualunque API compatibile con ``/chat/completions``
  (OpenAI, Azure OpenAI con endpoint compatibile, LM Studio, vLLM...). La chiave
  si legge solo dalla variabile d'ambiente ``RAG_OPENAI_API_KEY``.
* ``EchoLLM``: nessun modello; restituisce le frasi del contesto più vicine alla
  domanda. Serve per i test e per provare la pipeline senza LLM.

Le chiamate HTTP usano solo la libreria standard (``urllib``).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from rag_pdf_qa.config import Settings
from rag_pdf_qa.prompt import NOT_FOUND, Prompt
from rag_pdf_qa.text_utils import keywords


class LLMError(Exception):
    """Errore nella chiamata al modello linguistico."""


class LLM(ABC):
    """Interfaccia comune: riceve il prompt e restituisce il testo della risposta."""

    name: str

    @abstractmethod
    def generate(self, prompt: Prompt) -> str: ...


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LLMError(f"Errore HTTP {exc.code} da {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Impossibile contattare {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"Timeout dopo {timeout:.0f}s contattando {url}") from exc


class OllamaLLM(LLM):
    """Modello locale servito da Ollama sull'endpoint ``/api/chat``."""

    name = "ollama"

    def __init__(self, model: str, base_url: str, temperature: float = 0.1, timeout: float = 120) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: Prompt) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        }
        try:
            data = _post_json(f"{self.base_url}/api/chat", payload, {}, self.timeout)
        except LLMError as exc:
            raise LLMError(
                f"{exc}\nOllama è avviato? Prova 'ollama serve' e 'ollama pull {self.model}'."
            ) from exc
        try:
            return str(data["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Risposta inattesa da Ollama: {str(data)[:200]}") from exc


class OpenAICompatibleLLM(LLM):
    """Client per API compatibili con OpenAI Chat Completions."""

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        timeout: float = 120,
    ) -> None:
        if not api_key:
            raise LLMError("Manca la chiave: imposta RAG_OPENAI_API_KEY nel file .env.")
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: Prompt) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        data = _post_json(f"{self.base_url}/chat/completions", payload, headers, self.timeout)
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Risposta inattesa dall'API: {str(data)[:200]}") from exc


class EchoLLM(LLM):
    """"LLM" estrattivo: sceglie dal contesto le frasi con più parole in comune con la domanda.

    Non genera testo nuovo, quindi non può inventare: è utile come baseline e nei test.
    """

    name = "echo"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, prompt: Prompt) -> str:
        query = stems(keywords(prompt.question))
        candidates: list[tuple[int, int, str, str, set[str], set[str]]] = []
        seen: set[str] = set()
        for rank, result in enumerate(prompt.contexts):
            heading: set[str] = set()
            for position, sentence in enumerate(split_sentences(result.chunk.text)):
                if is_heading(sentence):
                    # il titolo non entra nella risposta, ma le sue parole danno contesto alle frasi
                    # della sezione ("Orari di apertura" → "La biblioteca è aperta dalle 9:00...")
                    heading = stems(keywords(sentence))
                    continue
                key = " ".join(sentence.lower().split())
                if key in seen or not is_answer_sentence(sentence):
                    continue  # duplicati (sovrapposizione tra chunk), domande e frammenti
                seen.add(key)
                candidates.append((rank, position, sentence, result.chunk.citation, stems(keywords(sentence)), heading))

        # le parole presenti in molte frasi ("costa", "transfer") pesano meno di quelle rare ("van")
        frequency: dict[str, int] = {}
        for *_, words, _heading in candidates:
            for word in words & query:
                frequency[word] = frequency.get(word, 0) + 1

        def weight(words: set[str]) -> float:
            return sum(1 / frequency.get(word, 1) for word in words)

        scored: list[tuple[float, int, int, str, str]] = []
        for rank, position, sentence, citation, words, heading in candidates:
            matched = query & words
            if matched:
                # bonus di sezione: la frase sta sotto un titolo che parla di ciò che si chiede
                score = weight(matched) + weight(query & heading)
                scored.append((score, -rank, -position, sentence, citation))
        if not scored:
            return NOT_FOUND
        scored.sort(reverse=True)
        # oltre alla migliore, solo frasi altrettanto pertinenti: niente "riempitivi"
        top = scored[0][0]
        best = [item for item in scored if item[0] >= top - 1e-9][: self.max_sentences]
        return " ".join(f"{sentence} [{citation}]" for *_, sentence, citation in best)


_SENTENCE_SPLIT = re.compile(r"(?<!Art\.)(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Frasi del testo: si divide sulla punteggiatura finale e sugli a capo tra paragrafi."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def stems(words: set[str]) -> set[str]:
    """Radice approssimata (prime 4 lettere) per confrontare singolare e plurale, maschile e
    femminile: "corsa"/"corse", "notturna"/"notturne", "apertura"/"aperta"."""
    return {w[:4] if len(w) > 4 else w for w in words}


def is_heading(sentence: str) -> bool:
    """Titolo di sezione: riga corta (meno di 12 parole) senza punteggiatura finale."""
    return not re.search(r"[.!?:;,]$", sentence) and len(sentence.split()) < 12 and sentence[:1].isupper()


def is_answer_sentence(sentence: str) -> bool:
    """Scarta i titoli (corti e senza punto finale) e i frammenti che cominciano a metà frase,
    creati dalla sovrapposizione tra chunk (prima lettera minuscola)."""
    first = sentence[:1]
    if first.isalpha() and first.islower():
        return False
    if sentence.endswith("?"):
        return False  # una domanda (es. il titolo di una FAQ) non è una risposta
    return bool(re.search(r"[.!?]$", sentence)) or len(sentence.split()) >= 12


def create_llm(settings: Settings) -> LLM:
    """Factory: crea il client LLM indicato in ``settings.llm``."""
    if settings.llm == "ollama":
        return OllamaLLM(settings.ollama_model, settings.ollama_url, settings.temperature, settings.timeout)
    if settings.llm == "openai":
        return OpenAICompatibleLLM(
            settings.openai_model,
            settings.openai_api_key,
            settings.openai_base_url,
            settings.temperature,
            settings.timeout,
        )
    if settings.llm == "echo":
        return EchoLLM()
    raise LLMError(f"LLM sconosciuto: {settings.llm!r} (usa 'ollama', 'openai' o 'echo')")
