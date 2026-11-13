from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from rag_pdf_qa import api  # noqa: E402
from rag_pdf_qa.pipeline import RagPipeline  # noqa: E402


def test_ask_endpoint(indexed_pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_pipeline", indexed_pipeline)
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/ask", json={"question": "Quanto dura il prestito dei libri?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] and "30 giorni" in body["answer"]
    assert body["sources"][0]["citation"].startswith("regolamento_biblioteca.pdf")


def test_ingest_blocks_path_traversal(indexed_pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_pipeline", indexed_pipeline)
    response = TestClient(api.app).post("/ingest", json={"path": "../../etc"})
    assert response.status_code == 400
