#!/usr/bin/env python
"""
Interactive CLI chat with the ERCOT Grid Agent.

Usage:
    python scripts/chat.py
    python scripts/chat.py --no-stream
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.rag.agent import GridAgent

console = Console()

BANNER = """
╔═══════════════════════════════════════════╗
║      ERCOT Grid Condition Q&A Agent       ║
║   Powered by open-source LLMs + RAG       ║
╚═══════════════════════════════════════════╝
Type 'quit' to exit. Type 'debug' to see retrieved context.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks to retrieve")
    args = parser.parse_args()

    console.print(BANNER, style="bold cyan")

    agent = GridAgent()
    console.print(f"[dim]Vector store loaded. {agent.store.count()} chunks indexed.[/dim]\n")

    if agent.store.count() == 0:
        console.print(
            "[yellow]⚠ No data in vector store. Run: python scripts/ingest.py[/yellow]\n"
        )

    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye.[/dim]")
            break

        # Debug mode: show retrieved chunks
        if question.lower() == "debug":
            q = console.input("[dim]Debug query:[/dim] ").strip()
            chunks = agent.get_context(q, top_k=args.top_k)
            for i, c in enumerate(chunks, 1):
                console.print(Panel(
                    f"[dim]{c['metadata']['title']}[/dim]\n\n{c['text']}",
                    title=f"Chunk {i} (distance: {c['distance']:.4f})",
                ))
            continue

        console.print("\n[bold blue]Agent:[/bold blue] ", end="")

        if args.no_stream:
            answer = agent.answer(question, top_k=args.top_k)
            console.print(Markdown(answer))
        else:
            full = []
            for chunk in agent.stream_answer(question, top_k=args.top_k):
                console.print(chunk, end="", highlight=False)
                full.append(chunk)
            console.print("\n")


if __name__ == "__main__":
    main()
