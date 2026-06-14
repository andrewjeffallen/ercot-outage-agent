#!/bin/bash
# Setup script to pull required Ollama models

echo "Pulling Ollama models..."
echo "This may take several minutes depending on your connection..."

# Pull the LLM model
echo "Pulling llama3.1:8b (4.9GB)..."
docker exec ercot-ollama ollama pull llama3.1:8b

# Pull the embedding model
echo "Pulling nomic-embed-text (274MB)..."
docker exec ercot-ollama ollama pull nomic-embed-text

echo ""
echo "Models pulled successfully!"
echo "You can now run the ingestion and chat scripts."
