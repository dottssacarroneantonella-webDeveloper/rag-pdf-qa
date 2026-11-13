"""Interfaccia a riga di comando.

Esempi::

    rag-pdf-qa ingest examples/
    rag-pdf-qa ask "Quali sono gli orari di apertura?"
    rag-pdf-qa ask "..." --llm echo --show-context
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from rag_pdf_qa.config import Settings
from rag_pdf_qa.embeddings import EmbedderError
from rag_pdf_qa.llm import LLMError
from rag_pdf_qa.loader import LoaderError
from rag_pdf_qa.pipeline import RagPipeline
from rag_pdf_qa.vector_store import VectorStoreError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-pdf-qa", description="Domande e risposte sui tuoi PDF con RAG."
    )
    parser.add_argument("--index-dir", help="cartella dell'indice (default: data/index)")
    parser.add_argument("-v", "--verbose", action="store_true", help="mostra i log dettagliati")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="indicizza un PDF o una cartella di PDF")
    ingest.add_argument("path", help="file PDF o cartella")
    ingest.add_argument("--embedder", choices=["sentence-transformers", "tfidf"])
    ingest.add_argument("--store", choices=["faiss", "numpy"])
    ingest.add_argument("--chunk-size", type=int)
    ingest.add_argument("--chunk-overlap", type=int)

    ask = sub.add_parser("ask", help="fai una domanda sui documenti indicizzati")
    ask.add_argument("question", help="la domanda, tra virgolette")
    ask.add_argument("--llm", choices=["ollama", "openai", "echo"])
    ask.add_argument("--model", help="nome del modello (Ollama o OpenAI)")
    ask.add_argument("-k", "--top-k", type=int)
    ask.add_argument("--show-context", action="store_true", help="mostra i chunk recuperati")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    settings = Settings.from_env()
    if args.index_dir:
        from pathlib import Path

        settings = settings.with_overrides(index_dir=Path(args.index_dir))

    try:
        if args.command == "ingest":
            settings = settings.with_overrides(
                embedder=args.embedder,
                store=args.store,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            report = RagPipeline(settings).ingest(args.path)
            print(
                f"Indicizzati {report.documents} documenti, {report.pages} pagine, "
                f"{report.chunks} chunk in {report.seconds:.1f}s -> {report.index_dir}"
            )
            return 0

        settings = settings.with_overrides(llm=args.llm, top_k=args.top_k)
        if args.model:
            field = "openai_model" if settings.llm == "openai" else "ollama_model"
            settings = settings.with_overrides(**{field: args.model})
        answer = RagPipeline(settings).ask(args.question)
        print(answer.text)
        if answer.answered and answer.citations():
            print("\nFonti: " + "; ".join(answer.citations()))
        if args.show_context:
            print("\nChunk recuperati:")
            for r in answer.sources:
                preview = r.chunk.text[:160].replace("\n", " ")
                print(f"  {r.score:.3f}  {r.chunk.citation}  {preview}...")
        return 0
    except (LoaderError, EmbedderError, VectorStoreError, LLMError, ValueError) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
