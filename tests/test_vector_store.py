from pathlib import Path

import numpy as np
import pytest

from rag_pdf_qa.embeddings import l2_normalize
from rag_pdf_qa.models import Chunk
from rag_pdf_qa.vector_store import NumpyStore, VectorStoreError, create_store, load_store

STORES = ["numpy"]
try:
    import faiss  # noqa: F401

    STORES.append("faiss")
except ImportError:
    pass


def sample() -> tuple[list[Chunk], np.ndarray]:
    chunks = [Chunk(id=f"d.pdf:p{i}:c1", source="d.pdf", page=i, text=f"testo {i}") for i in range(1, 4)]
    vectors = l2_normalize(np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32))
    return chunks, vectors


@pytest.mark.parametrize("kind", STORES)
def test_search_orders_by_cosine_similarity(kind: str) -> None:
    store = create_store(kind)
    store.add(*sample())
    results = store.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)
    assert [r.chunk.page for r in results] == [1, 3]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].score == pytest.approx(0.7071, abs=1e-3)


@pytest.mark.parametrize("kind", STORES)
def test_save_and_load(kind: str, tmp_path: Path) -> None:
    store = create_store(kind)
    store.add(*sample())
    store.save(tmp_path)
    loaded = load_store(tmp_path)
    assert loaded.name == kind and len(loaded) == 3
    assert loaded.search(np.array([0, 1, 0], dtype=np.float32), 1)[0].chunk.page == 2


def test_empty_store_and_mismatch() -> None:
    store = NumpyStore()
    assert store.search(np.array([1.0, 0.0]), 3) == []
    chunks, vectors = sample()
    with pytest.raises(VectorStoreError):
        store.add(chunks[:2], vectors)


def test_unknown_store_and_missing_index(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreError):
        create_store("chroma")
    with pytest.raises(VectorStoreError, match="ingest"):
        load_store(tmp_path)
