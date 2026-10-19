<div align="center">

<img src="./static/image/mirofish-offline-banner.png" alt="MiroFish Offline" width="100%"/>

# MiroFish-Offline

**English-language fork of [MiroFish](https://github.com/666ghj/MiroFish) — local-first, with optional Groq routing for supported stages.**

*A multi-agent simulation engine for exploring public opinion, market sentiment, and social dynamics, using local models where practical.*

[![GitHub Stars](https://img.shields.io/github/stars/nikmcfly/MiroFish-Offline?style=flat-square&color=DAA520)](https://github.com/nikmcfly/MiroFish-Offline/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/nikmcfly/MiroFish-Offline?style=flat-square)](https://github.com/nikmcfly/MiroFish-Offline/network)
[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](./LICENSE)

</div>

## What is this?

MiroFish is a multi-agent simulation engine: upload any document (press release, policy draft, financial report), and it generates hundreds of AI agents with unique personalities that simulate the public reaction on social media. Posts, arguments, opinion shifts — hour by hour.

The [original MiroFish](https://github.com/666ghj/MiroFish) uses a different UI language and cloud services. This fork provides an **English UI** and replaces Zep Cloud graph storage with local Neo4j:

| Original project | MiroFish-Offline |
|---|---|
| Original-language UI | **English UI** |
| Zep Cloud (graph memory) | **Neo4j Community Edition 5.15** |
| DashScope / OpenAI API (LLM) | **Ollama**, with optional **Groq** routing |
| Zep Cloud embeddings | **nomic-embed-text** via Ollama |
| Cloud graph services | **Local Neo4j graph storage** |

## Workflow

1. **Graph Build** — Extracts entities (people, companies, events) and relationships from your document. Ontology analysis samples up to 10,000 characters for model context; the full document is retained for graph construction. Builds a knowledge graph with individual and group memory via Neo4j.
2. **Env Setup** — Generates hundreds of agent personas, each with unique personality, opinion bias, reaction speed, influence level, and memory of past events.
3. **Simulation** — Agents interact on simulated social platforms: posting, replying, arguing, shifting opinions. The system tracks sentiment evolution, topic propagation, and influence dynamics in real time.
4. **Report** — A ReportAgent analyzes the simulated environment, can interview simulation agents, searches the knowledge graph for evidence, and generates a structured analysis.
5. **Interaction** — Chat with any agent from the simulated world. Ask them why they posted what they posted. Full memory and personality persists.

## Screenshot

<div align="center">
<img src="./static/image/mirofish-offline-screenshot.jpg" alt="MiroFish Offline — English UI" width="100%"/>
</div>

## Quick Start

### Prerequisites

- Docker & Docker Compose (recommended), **or**
- Python 3.11+, Node.js 18+, Neo4j 5.15+, Ollama

### Option A: Docker (easiest)

```bash
git clone https://github.com/nikmcfly/MiroFish-Offline.git
cd MiroFish-Offline
cp .env.example .env

# Start the app and Neo4j (Ollama runs natively on your machine)
docker compose up -d

# Pull the required models into your local Ollama
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

Make sure Ollama is running and listening on port 11434. Open `http://localhost:3000`.

### Option B: Manual

**1. Start Neo4j**

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/mirofish \
  neo4j:5.15-community
```

**2. Start Ollama on your machine & pull models**

```bash
ollama serve &
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

**3. Configure & run backend**

```bash
cp .env.example .env
# Edit .env if your Neo4j/Ollama are on non-default ports

cd backend
pip install -r requirements.txt
python run.py
```

**4. Run frontend**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Configuration

All settings are in `.env` (copy from `.env.example`):

```bash
# Docker: host.docker.internal points to Ollama running on the host machine.
LLM_API_KEY=ollama
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL_NAME=qwen3.5:9b

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mirofish

# Embeddings
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://host.docker.internal:11434

# Optional provider routing. Groq is used only for stages with a configured model.
GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_ONTOLOGY_MODEL=
GROQ_SIMULATION_MODEL=
GROQ_GRAPH_MODEL=
GROQ_GRAPH_FREE_TIER_CONFIRMED=false
GROQ_GRAPH_TIMEOUT_SECONDS=90
```

Groq ontology routing activates when `GROQ_ONTOLOGY_MODEL` is set and requires `GROQ_API_KEY`; Groq simulation routing is configured independently with `GROQ_SIMULATION_MODEL`. Graph extraction has its own `GROQ_GRAPH_MODEL` and requires Groq to be configured. Before enabling it, check your account's model access, pricing, and quotas, then set `GROQ_GRAPH_FREE_TIER_CONFIRMED=true` only if you have verified the selected model is covered by your free tier. In my testing from India, Alibaba/DashScope was not available in my setup, while a Groq-hosted Qwen model worked on the free tier at the time. Provider availability and free-tier limits vary by account and can change. Graph extraction uses bounded 2,000-character chunks with 200-character overlap; local Ollama handles embeddings and can run the other configured stages. For native Windows backend runs, use `http://localhost:11434` instead of `host.docker.internal`. Ontology analysis samples up to 10,000 characters; the original document remains intact for graph construction.

## Architecture

This fork introduces a clean abstraction layer between the application and the graph database:

```
┌─────────────────────────────────────────┐
│              Flask API                   │
│  graph.py  simulation.py  report.py     │
└──────────────┬──────────────────────────┘
               │ app.extensions['neo4j_storage']
┌──────────────▼──────────────────────────┐
│           Service Layer                  │
│  EntityReader  GraphToolsService         │
│  GraphMemoryUpdater  ReportAgent         │
└──────────────┬──────────────────────────┘
               │ storage: GraphStorage
┌──────────────▼──────────────────────────┐
│         GraphStorage (abstract)          │
│              │                            │
│    ┌─────────▼─────────┐                │
│    │   Neo4jStorage     │                │
│    │  ┌───────────────┐ │                │
│    │  │ EmbeddingService│ ← Ollama       │
│    │  │ NERExtractor   │ ← Configured LLM│
│    │  │ SearchService  │ ← Hybrid search │
│    │  └───────────────┘ │                │
│    └───────────────────┘                │
└─────────────────────────────────────────┘
               │
        ┌──────▼──────┐
        │  Neo4j CE   │
        │  5.15       │
        └─────────────┘
```

**Key design decisions:**

- `GraphStorage` is an abstract interface — swap Neo4j for any other graph DB by implementing one class
- Dependency injection via Flask `app.extensions` — no global singletons
- Hybrid search: 0.7 × vector similarity + 0.3 × BM25 keyword search
- Synchronous NER/RE extraction via the configured provider, using bounded overlapping chunks (replaces Zep's async episodes)
- All original dataclasses and LLM tools (InsightForge, Panorama, Agent Interviews) preserved

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| VRAM (GPU) | 10 GB (14b model) | 24 GB (32b model) |
| Disk | 20 GB | 50 GB |
| CPU | 4 cores | 8+ cores |

CPU-only mode works but is significantly slower for LLM inference. For lighter setups, use a smaller model supported by your Ollama installation.

## Testing Notes

I tested the workflow with Ollama `qwen3.5:9b` locally and a Groq-hosted Qwen 27B model. Both paths can be slow: the application runs several model-backed stages, and local inference depends heavily on available hardware. The larger Groq model produced thorough results in my tests, but generation took time and ran into provider token limits.

Longer runs use bounded document chunks with overlap, smaller report prompts, and compacted retrieval context to reduce request sizes. These strategies help with provider limits, but do not remove rate limits or guarantee fast generation. Allow time for graph construction and report generation.

## Use Cases

- **PR crisis testing** — simulate the public reaction to a press release before publishing
- **Trading signal generation** — feed financial news and observe simulated market sentiment
- **Policy impact analysis** — test draft regulations against simulated public response
- **Creative experiments** — explore how simulated agents respond to fictional scenarios and story prompts

## License

AGPL-3.0 — same as the original MiroFish project. See [LICENSE](./LICENSE).

## Credits & Attribution

This is a modified fork of [MiroFish](https://github.com/666ghj/MiroFish) by [666ghj](https://github.com/666ghj), originally supported by [Shanda Group](https://www.shanda.com/). The simulation engine is powered by [OASIS](https://github.com/camel-ai/oasis) from the CAMEL-AI team.

**Modifications in this fork:**
- Backend migrated from Zep Cloud to local Neo4j CE 5.15 + Ollama
- Frontend translated into English
- All Zep references replaced with Neo4j across the UI
- Rebranded to MiroFish Offline
