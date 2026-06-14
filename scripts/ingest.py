#!/usr/bin/env python
"""
Ingest ERCOT data into ChromaDB.

Usage:
    python scripts/ingest.py           # uses mock data (safe for local testing)
    python scripts/ingest.py --live    # hits real ERCOT endpoints
    python scripts/ingest.py --reset   # wipe DB then ingest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import track

from src.ingestion.ercot import fetch_grid_notices, fetch_mock_documents
from src.rag.store import VectorStore

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ingest ERCOT data into ChromaDB")
    parser.add_argument("--live", action="store_true", help="Fetch from real ERCOT endpoints")
    parser.add_argument("--reset", action="store_true", help="Reset vector store before ingesting")
    parser.add_argument("--limit", type=int, default=50, help="Max documents to fetch (live mode)")
    args = parser.parse_args()

    store = VectorStore()

    if args.reset:
        console.print("[yellow]Resetting vector store...[/yellow]")
        store.reset()

    console.print(f"[blue]Current documents in store: {store.count()}[/blue]")

    if args.live:
        console.print("[green]Fetching live ERCOT grid notices...[/green]")
        docs = fetch_grid_notices(limit=args.limit)
    else:
        console.print("[cyan]Using mock ERCOT data (pass --live for real data)[/cyan]")
        docs = fetch_mock_documents()

    console.print(f"Fetched [bold]{len(docs)}[/bold] documents. Embedding and storing...")

    chunks_added = store.add_documents(docs)

    console.print(f"[green]✓ Done. Added {chunks_added} chunks. Total in store: {store.count()}[/green]")


if __name__ == "__main__":
    main()
