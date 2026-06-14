# ERCOT Grid Condition Q&A Agent

RAG pipeline over live ERCOT operational data (outages, grid notices, curtailments)
powered by open-source LLMs running locally via Ollama.

## Architecture

```
ERCOT Public API (XML/CSV)
        │
        ▼
   src/ingestion/          ← fetch, parse, chunk ERCOT data
        │
        ▼
   src/rag/                ← embed chunks → ChromaDB, retrieval
        │
        ▼
   src/inference/          ← Ollama client (local LLM)
        │
        ▼
   Gradio UI / CLI
```

## Stack

| Layer | Technology |
|---|---|
| LLM | Ollama `llama3.1:8b` |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector DB | ChromaDB (local persistent) |
| Data | ERCOT via gridstatus library |
| Orchestration | Docker Compose (optional) |

## Quickstart

### Option 1: Docker (Recommended - with memory controls)

```bash
# 1. Copy and fill in your env vars
cp .env.example .env

# 2. Start the services (Ollama + Agent)
docker-compose up -d ollama agent

# 3. Pull Ollama models (first time only)
./scripts/setup_ollama.sh

# 4. Ingest ERCOT data
docker-compose --profile ingest up ingest

# 5. Run the chat agent
docker exec -it ercot-agent uv run python scripts/chat.py
```

**Memory limits** (adjust in `docker-compose.yml` based on your system):
- Ollama: 8GB limit / 4GB reserved
- Agent: 4GB limit / 2GB reserved
- Ingest: 2GB limit / 1GB reserved

### Option 2: Local (without Docker)

```bash
# 1. Install uv (if not already installed)
# https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install deps and create virtual environment
uv sync

# 3. Copy and fill in your env vars
cp .env.example .env

# 4. Install Ollama and pull model (local dev)
# https://ollama.com/download
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 5. Ingest ERCOT data
uv run python scripts/ingest.py --live

# 6. Run the agent
uv run python scripts/chat.py
```

## Project Status

- ✅ Local RAG pipeline with Ollama
- ✅ Docker deployment with memory controls
- ✅ Live ERCOT data ingestion via gridstatus
- ✅ CLI chat interface with streaming
- 🚧 Gradio web UI (planned)
