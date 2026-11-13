from pathlib import Path

import numpy as np
import pytest

from rag_pdf_qa.embeddings import EmbedderError, TfidfEmbedder, create_embedder, load_embedder


def test_tfidf_vectors_are_normalized_and_semantic_enough() -> None:
    embedder = TfidfEmbedder()
    texts = ["orari di apertura della biblioteca", "durata del prestito dei libri", "rete wifi gratuita"]
    embedder.fit(texts)
    vectors = embedder.embed(texts)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    query = embedder.embed_query("quanto dura il prestito?")
    assert int(np.argmax(vectors @ query)) == 1


def test_tfidf_save_and_load(tmp_path: Path) -> None:
    embedder = TfidfEmbedder()
    embedder.fit(["uno due tre", "quattro cinque"])
    embedder.save(tmp_path)
    loaded = load_embedder(tmp_path)
    assert np.allclose(loaded.embed(["uno due"]), embedder.embed(["uno due"]))


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(EmbedderError):
        TfidfEmbedder().embed(["non adattato"])
    with pytest.raises(EmbedderError):
        create_embedder("word2vec")
    with pytest.raises(EmbedderError, match="ingest"):
        load_embedder(tmp_path)
