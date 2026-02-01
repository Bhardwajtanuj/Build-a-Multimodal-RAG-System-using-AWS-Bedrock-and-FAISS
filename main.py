"""CLI for the Multimodal RAG System."""

import argparse
from pathlib import Path

from src.embeddings import BedrockEmbeddings
from src.ingest import ingest_all, ingest_documents, ingest_images
from src.rag import MultimodalRAG
from src.vector_store import FAISSVectorStore
from src.config import DOCUMENTS_DIR, IMAGES_DIR


def cmd_ingest(args):
    """Ingest documents and/or images."""
    embeddings = BedrockEmbeddings()
    vector_store = FAISSVectorStore(dimension=embeddings.dimension)
    vector_store.load()

    doc_count = 0
    img_count = 0

    if args.documents or args.all:
        paths = [Path(p) for p in args.documents] if args.documents else []
        doc_count = ingest_documents(
            paths=paths or None,
            directory=DOCUMENTS_DIR if args.all else None,
            embeddings=embeddings,
            vector_store=vector_store,
        )

    if args.images or args.all:
        paths = [Path(p) for p in args.images] if args.images else None
        img_count = ingest_images(
            paths=paths,
            directory=IMAGES_DIR if args.all else None,
            embeddings=embeddings,
            vector_store=vector_store,
        )

    print(f"Ingested: {doc_count} text chunks, {img_count} images.")
    print(f"Total vectors: {vector_store.count}")


def cmd_query(args):
    """Run a RAG query."""
    rag = MultimodalRAG()
    if not rag.load_index():
        print("Error: No index found. Run 'ingest' first.")
        return

    question = args.question or "What is in this image?"
    query_image = Path(args.image) if args.image else None

    print("Retrieving and generating...")
    answer, docs = rag.query(question, query_image_path=query_image, top_k=args.top_k)

    print("\n--- Answer ---")
    print(answer)
    print("\n--- Retrieved Sources ---")
    for i, d in enumerate(docs, 1):
        print(f"\n{i}. [{d.get('type')}] {d.get('source', 'N/A')}")
        print(d.get("content", "")[:200] + "..." if len(d.get("content", "")) > 200 else d.get("content", ""))


def cmd_app(args):
    """Launch Streamlit app."""
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(args.port),
        "--server.headless", "true",
    ])


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG System (AWS Bedrock + FAISS)")
    sub = parser.add_subparsers(dest="command", required=True)

    # Ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents and images")
    p_ingest.add_argument("--all", action="store_true", help="Ingest from data/documents and data/images")
    p_ingest.add_argument("--documents", nargs="*", help="Document file paths")
    p_ingest.add_argument("--images", nargs="*", help="Image file paths")
    p_ingest.set_defaults(func=cmd_ingest)

    # Query
    p_query = sub.add_parser("query", help="Run a RAG query")
    p_query.add_argument("question", nargs="?", help="Question to ask")
    p_query.add_argument("--image", help="Query image path (for image-based search)")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    p_query.set_defaults(func=cmd_query)

    # App
    p_app = sub.add_parser("app", help="Launch Streamlit app")
    p_app.add_argument("--port", type=int, default=8501, help="Port for Streamlit")
    p_app.set_defaults(func=cmd_app)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
