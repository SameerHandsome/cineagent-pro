# 🎬 CineAgent Pro

AI-powered film pre-production intelligence. Paste a logline and get a complete Pre-Production Intelligence Report — script analysis, budget estimates, casting suggestions, and market intelligence — in one conversation. Follow-up questions are answered with context from everything already discussed.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture Overview](#architecture-overview)
- [How the Pipeline Works](#how-the-pipeline-works)
- [File-by-File Breakdown](#file-by-file-breakdown)
- [Required API Keys](#required-api-keys)
- [Local Setup](#local-setup)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [Kubernetes Deployment](#kubernetes-deployment)
- [CI/CD](#cicd-github-actions)
- [Common Issues](#common-issues)

---

## What It Does

You type a logline like:

> *"A disgraced CIA operative tracks a ghost network selling classified identities — shot in Geneva, Macau, and Lagos for $28M."*

CineAgent Pro routes that to four specialist AI agents running in parallel. Within seconds you receive a structured report covering:

- **Script analysis** — genres, tone, structural complexity, characters, themes, and budget flags extracted from your concept
- **Budget estimate** — department-level line items using live SAG-AFTRA and IATSE rates, with a tier label (micro / indie / mid / a-list)
- **Casting suggestions** — three ranked actors per principal character, filtered by budget tier and flagged for recent controversy
- **Market intelligence** — five real comparable films with ROI, distribution recommendation (theatrical / streaming / hybrid), top platform, and optimal release window

You can then ask follow-up questions and the agents answer from the context already in scope — no need to re-paste your concept.

---

## Architecture Overview

```
User message
    │
    ▼
context_assembly ──► Redis: last 5 messages  (fallback: PostgreSQL)
                     Qdrant: past project summaries for this user (RAG)
    │
    ▼
orchestrator ──► classifies intent (full_analysis | refine | budget_only | ...)
                 decides which agents to activate and whether this is a
                 new project or a follow-up refinement
    │
    ├──── script_analyst   ──► MCP tools: parse_screenplay, extract_characters,
    │     (70b model)           analyze_themes, identify_key_scenes
    │                           Playwright MCP: scrape IMDb comps
    │
    │     ┌──────────────────────────────────────────────────────┐
    │     │  Parallel fan-out (staggered 0s / 1s / 2s to avoid  │
    │     │  Groq TPM burst limit)                               │
    │     └──────────────────────────────────────────────────────┘
    │
    ├──── budget_planner   ──► MCP tools: calculate_budget_line, get_union_rate_from_db
    │     (8b model)           Brave Search MCP: live SAG/IATSE rates
    │                           Filesystem MCP: user-uploaded budget templates
    │
    ├──── casting_director ──► MCP tools: search_casting_db, get_casting_preferences
    │     (8b model)           Brave Search MCP: actor career news
    │
    └──── market_intel     ──► MCP tools: get_market_comps_from_db, get_streaming_landscape
          (70b model)          Playwright MCP: live Box Office Mojo
                               Brave Search MCP: streaming acquisition news
    │
    ▼
join barrier ──► waits for all parallel branches
    │
    ▼
synthesizer ──► assembles all agent outputs into the final report
    │            MODE A (new project): full Pre-Production Intelligence Report
    │            MODE B (follow-up):   focused answer to the specific question
    ▼
SSE stream ──► report chunks streamed to the browser in real time
    │
    ▼
Celery worker (background, after response is sent)
    ──► Groq summarises the session in 2-3 sentences
    ──► Qdrant indexes: session summary + extracted user preferences
        (makes the next session smarter via RAG)
```

### Prompt Assembly Order (every agent)

Every agent assembles its context in this fixed order:

```
[1] System prompt    — role, tool call order, output format rules
[2] RAG context      — Qdrant: past session summaries for this user
[3] Session history  — last 5 messages (Redis, fallback PostgreSQL)
[4] User query       — the current message
[5] agent_scratchpad — MCP tool call / result pairs (grows during loop)
```

The scratchpad is always last. This is a hard constraint — Groq's llama models decide "call another tool vs. emit final text" based on what comes last in the message sequence. Putting anything after the scratchpad confuses the model.

### Model Routing

Two separate TPM buckets on Groq free tier:

| Agent | Model | TPM bucket | Reason |
|---|---|---|---|
| orchestrator | llama-4-scout (8b) | 30K | JSON classification only |
| script_analyst | llama-3.3-70b-versatile | 12K | Nuanced extraction |
| budget_planner | llama-4-scout (8b) | 30K | Deterministic math |
| casting_director | llama-4-scout (8b) | 30K | DB lookup + brief notes |
| market_intel | llama-3.3-70b-versatile | 12K | Trend reasoning |
| synthesizer | llama-4-scout (8b) | 30K | Report assembly |

---

## How the Pipeline Works

### Turn 1 — New Film Concept

1. `context_assembly` loads Redis history (empty on first turn) and Qdrant RAG context for this user.
2. `orchestrator` classifies intent as `full_analysis`, activates all four agents.
3. `script_analyst` runs first (sequential). It uses MCP tools to parse the concept and scrape IMDb comps, then outputs structured data in labelled plain text (not JSON — Groq rejects JSON as a final answer when tools are bound).
4. `budget_planner`, `casting_director`, `market_intel` run in parallel with a 0s / 1s / 2s stagger. Each agent uses its own MCP tools, reads the script analyst's output from shared LangGraph state, and also outputs plain text.
5. `synthesizer` assembles everything into a markdown report and streams it to the browser via SSE.
6. Celery worker runs in the background, summarises the session, and indexes it to Qdrant.

### Turn 2+ — Follow-Up Questions

1. `context_assembly` loads the prior conversation from Redis.
2. `orchestrator` detects this is a follow-up (`intent=refine`) and activates only the relevant agent(s). For example, "what's the marketing strategy?" activates only `market_intel`.
3. All agents suppress their MCP tools on follow-up turns — tools are only useful when there is fresh material to parse or live data to fetch. On follow-ups, tools would cause the model to hallucinate search queries unrelated to the actual question.
4. `synthesizer` receives MODE B context — only the activated agent's output plus the conversation history — and produces a focused answer without regenerating the full report.

---

## File-by-File Breakdown

### `backend/config.py`
Single source of truth for all environment variables. Uses `pydantic-settings` so every setting is type-validated at startup. All other modules import `settings` from here — there are no scattered `os.getenv()` calls.

### `backend/main.py`
FastAPI application entry point. The `lifespan` context manager runs the startup sequence: configure LangSmith tracing → create PostgreSQL tables → ensure Qdrant collection exists → load all MCP tools → initialise the tool registry. Registers all routers and CORS middleware.

### `backend/agents/_base.py`
The shared agentic loop used by every specialist agent. Handles: building the LLM instance with the correct model and token budget, binding tools, executing the tool call / result loop (max 4 iterations to avoid TPM burst), type-coercing tool arguments (Groq sometimes emits integers as strings), and silently recovering from malformed tool call errors (Groq 400 `tool_use_failed`) by retrying the same call without tools bound.

### `backend/agents/orchestrator.py`
Lightweight intent classifier. Reads the user message and session history, then outputs a JSON object with `intent` and `active_agents`. Uses an 8b model since it does nothing but classify. Includes a post-parse signal detector that overrides `full_analysis` to `refine` when the model misclassifies a short follow-up (common with 8b on queries that have no film-concept signal).

### `backend/agents/script_analyst.py`
Parses a film concept into structured production data: genres, tone, structural complexity, character list, themes, budget flags, and IMDb comparable films. Uses the 70b model for higher extraction quality. Outputs labelled plain text (not JSON) to avoid Groq's tool-use parser intercepting the final answer. `_parse_labelled_output()` converts the text into a typed dict that the downstream agents read from LangGraph state.

### `backend/agents/budget_planner.py`
Produces a department-level budget estimate using live SAG-AFTRA and IATSE rates fetched via Brave Search, cross-checked against the local union rate DB. Reads `structural_complexity` and `budget_flags` from script analyst state to estimate shoot days. Labels the project as micro / indie / mid / a-list.

### `backend/agents/casting_director.py`
Suggests three ranked actors per principal character, filtered by budget tier, using the internal casting DB and Brave Search for career momentum. Flags any recent controversy found in search results.

### `backend/agents/market_intel.py`
Analyses commercial viability: pulls five comparable films from the seeded dataset, navigates Box Office Mojo for live data, fetches streaming acquisition news via Brave Search, and produces a distribution recommendation with a specific platform and release window.

### `backend/agents/orchestrator.py`
*(see above)*

### `backend/graph/state.py`
`CineAgentState` TypedDict. The single shared state object that flows through the entire LangGraph workflow. Every agent reads from it and writes to it. `total=False` means all keys are optional — agents return only the keys they update.

### `backend/graph/nodes.py`
Contains `context_assembly_node` (loads Redis history + Qdrant RAG before any agent runs) and `synthesizer_node` (assembles all agent outputs into the final report). The synthesizer builds different context depending on `intent`: MODE A (full report) injects all four agent sections; MODE B (follow-up) injects only the activated agent's output plus the conversation history, preventing the model from defaulting to a full-report format.

### `backend/graph/workflow.py`
Defines the LangGraph `StateGraph`. The graph topology is: `context_assembly → orchestrator → script_analyst → [budget_planner | casting_director | market_intel] (parallel) → join → synthesizer`. Each parallel agent is wrapped in a staggered coroutine (0s / 1s / 2s delays) to spread Groq API calls across TPM windows.

### `backend/prompt/templates.py`
All prompts live here — system prompts, RAG context slots, session history slots, output format specifications. The output format for tool-using agents is labelled plain text (not JSON). The synthesizer is the only agent that uses markdown because it has no tools bound and no risk of Groq intercepting its output as a tool call.

### `backend/api/chat.py`
`POST /api/chat` — validates the JWT, checks rate limit, resolves or creates the session, persists the user message to Redis and PostgreSQL, streams the workflow output as SSE events (`[SESSION:...]`, `[AGENT:...]`, report chunks, `[DONE]`), then fires the background Celery indexing task. Also handles session CRUD: list, create, delete, patch title, get messages.

### `backend/api/health.py`
Single `GET /health` endpoint. Used by the CI pipeline, Docker health checks, and Kubernetes liveness probes.

### `backend/api/profile.py`
`GET /api/profile` — returns the current user's email, username, creation date, and session count.

### `backend/auth/jwt_handler.py`
Creates and decodes JWTs signed with `SECRET_KEY`. Tokens expire after `ACCESS_TOKEN_EXPIRE_MINUTES` (default 60).

### `backend/auth/router.py`
Email/password register and login, plus GitHub OAuth (redirect → callback → JWT → frontend redirect). Password hashing with `bcrypt` via `passlib`.

### `backend/database/connection.py`
Async SQLAlchemy engine pointed at Neon PostgreSQL. Connection pool: 5 connections, 10 overflow, pre-ping enabled. `init_db()` creates all tables on startup (idempotent).

### `backend/database/models.py`
Three ORM models: `User` (email, username, hashed_password, github_id), `Session` (title, user FK), `Message` (role, content, session FK).

### `backend/database/crud.py`
All database operations. Key design decisions: `create_session` accepts an optional `session_id` so the UUID generated in `chat.py` is used in both Redis and PostgreSQL (previously a different UUID was generated internally, breaking history lookup). `delete_session` uses raw SQL Core (not ORM) to delete messages first, avoiding a SQLAlchemy identity map bug where orphaned NULL-session_id rows would be flushed as a NOT NULL constraint violation.

### `backend/database/schemas.py`
Pydantic response models for the API layer (`UserOut`, `SessionOut`, `MessageOut`). Separate from ORM models to avoid leaking internal fields.

### `backend/cache/redis_client.py`
Upstash Redis client (asyncio). Provides: session message history (last 20 messages, 5-day TTL), tool result caching (6-hour TTL to avoid re-fetching the same Brave Search / Playwright results), and rate limiting (sliding window, default 15 requests per 60 seconds).

### `backend/rag/embedder.py`
Generates 384-dimension sentence embeddings using `all-MiniLM-L6-v2` via `sentence-transformers`. Runs locally on CPU, no API cost. The model is cached via `@lru_cache` so it is loaded from disk only once per process.

### `backend/rag/retriever.py`
Retrieves vectors from Qdrant. Every query includes a hard `tenant_id == user_id` filter applied at the Qdrant storage layer — not the application layer. This ensures User A's vectors are never visible to User B even if there is a bug in the application code. Uses `query_points()` (the API introduced in qdrant-client 1.7, replacing the removed `.search()`).

### `backend/rag/indexer.py`
Upserts vectors into Qdrant with full multi-tenancy metadata: `tenant_id`, `doc_type` (`session_summary` or `user_preference`), `topic` (genre slug), `session_id`, `title`, `summary`. Creates payload indexes on those fields at collection creation time so Qdrant filters at the storage layer rather than doing a full scan.

### `backend/worker/celery_app.py`
Celery application configured with CloudAMQP (RabbitMQ) as the broker. Fire-and-forget — no result backend needed since background tasks do not return values to the HTTP request that triggered them.

### `backend/worker/tasks.py`
`trigger_background_indexing` runs after every successful chat response. It calls Groq to summarise the session in 2-3 sentences, then indexes the summary to Qdrant. It also extracts a user preference statement (e.g. "Prefers streaming distribution for indie sci-fi") and indexes that separately so the retriever can fetch preferences independently of session summaries.

### `mcp_clients/loader.py`
Loads all MCP tool servers on startup: the local FastMCP server (HTTP transport) and four remote servers (Brave Search via SSE, Playwright via stdio subprocess, Filesystem via stdio subprocess, GitHub via SSE).

### `mcp_clients/tool_registry.py`
Slices the merged tool list per agent. Each agent calls `get_registry().script_tools` (or `budget_tools`, etc.) rather than receiving the full tool list.

### `mcp_server/server.py`
Local FastMCP server running on port 8001. Registers four tool groups: script tools (`parse_screenplay`, `extract_characters`, `analyze_themes`, `identify_key_scenes`), budget tools (`calculate_budget_line`, `get_union_rate_from_db`), casting tools (`search_casting_db`, `get_casting_preferences`), market tools (`get_market_comps_from_db`, `get_streaming_landscape`).

### `tests/conftest.py`
Shared pytest fixtures. All external services are mocked here: Groq (AsyncMock returning configurable responses), Qdrant (AsyncMock with empty `.points`), Redis (AsyncMock), Celery tasks (MagicMock `.delay`), sentence-transformers (returns a deterministic 384-float vector), and the MCP tool registry (returns empty tool lists). `autouse=True` on the critical mocks means no test accidentally hits a real service.

---

## Required API Keys

| Key | Where to get it | Free tier |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | ✅ Generous free |
| `DATABASE_URL` | [neon.tech](https://neon.tech) → Pooled connection string | ✅ Free |
| `UPSTASH_REDIS_REST_URL` + `TOKEN` | [console.upstash.com](https://console.upstash.com) → REST API | ✅ Free |
| `REDIS_URL` | Same Upstash instance → Redis URL | ✅ Free |
| `CELERY_BROKER_URL` | [cloudamqp.com](https://cloudamqp.com) → Little Lemur plan | ✅ Free |
| `QDRANT_URL` + `QDRANT_API_KEY` | [cloud.qdrant.io](https://cloud.qdrant.io) | ✅ Free 1GB |
| `GITHUB_CLIENT_ID` + `SECRET` | [github.com/settings/developers](https://github.com/settings/developers) → OAuth Apps | ✅ Free |
| `GITHUB_TOKEN` | [github.com/settings/tokens](https://github.com/settings/tokens) → Fine-grained PAT | ✅ Free |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | ✅ 1000 req/month |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) | ✅ Free |
| `SECRET_KEY` | Generate yourself (see below) | — |

**Generate SECRET_KEY:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**GitHub OAuth callback URL** (set when creating the OAuth App):
```
http://localhost:8000/auth/github/callback
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm (for Playwright MCP and Filesystem MCP)
- Git

### 1. Clone and create virtual environment

```bash
git clone https://github.com/SameerHandsome/cineagent-pro.git
cd cineagent-pro

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Node dependencies

```bash
# Playwright MCP — browser automation for IMDb / Box Office Mojo scraping
npx playwright install --with-deps chromium

# Filesystem MCP is installed on-demand by npx — no manual step needed
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in every value. Minimum required:

```
SECRET_KEY=<generated>
GROQ_API_KEY=gsk_...
DATABASE_URL=postgresql+asyncpg://...
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...
REDIS_URL=rediss://...
CELERY_BROKER_URL=amqps://...
QDRANT_URL=https://...
QDRANT_API_KEY=...
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
```

Optional (agents degrade gracefully without these):

```
TAVILY_API_KEY=...        # live web search
GITHUB_TOKEN=...          # GitHub MCP
LANGCHAIN_API_KEY=...     # LangSmith tracing
```

### 5. Create the uploads directory

```bash
mkdir -p uploads
```

### 6. Start all services (4 terminals)

**Terminal 1 — Local MCP server:**
```bash
source venv/bin/activate
python mcp_server/server.py
# FastMCP server running on http://0.0.0.0:8001
```

**Terminal 2 — FastAPI backend:**
```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# Application startup complete.
```

**Terminal 3 — Celery worker:**
```bash
source venv/bin/activate
celery -A backend.worker.celery_app worker --loglevel=info --concurrency=2
```

**Terminal 4 — Frontend:**
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000/index.html
```

### 7. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"CineAgent Pro"}

curl http://localhost:8001/health
# FastMCP health response
```

---

## Running Tests

All external services are mocked. No real API keys needed.

```bash
# Install test dependencies
pip install pytest pytest-asyncio aiosqlite pytest-mock

# All tests
pytest

# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific file
pytest tests/unit/test_agents.py -v

# With coverage
pip install pytest-cov
pytest --cov=backend --cov=mcp_clients --cov-report=term-missing
```

**Code quality checks (same as CI):**
```bash
pip install black ruff

black --check --diff .   # check formatting
black .                  # auto-fix

ruff check .             # lint check
ruff check --fix .       # auto-fix
```

---

## Docker

```bash
# Build and run everything
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up

# Build image manually
docker build -f docker/Dockerfile -t cineagent-pro:latest .

# Run backend
docker run -p 8000:8000 --env-file .env cineagent-pro:latest

# Run Celery worker
docker run --env-file .env cineagent-pro:latest \
  celery -A backend.worker.celery_app worker --loglevel=info

# Run MCP server
docker run -p 8001:8001 --env-file .env cineagent-pro:latest \
  python mcp_server/server.py
```

---

## Kubernetes Deployment

```bash
# Encode secrets
echo -n "your-value" | base64

# Apply in order
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap-secret.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Verify
kubectl get pods -n cineagent

# Force pull latest image after a push
kubectl rollout restart deployment/cineagent-backend -n cineagent
kubectl rollout restart deployment/cineagent-worker  -n cineagent
kubectl rollout restart deployment/mcp-server        -n cineagent
```

---

## CI/CD (GitHub Actions)

```
Push to main / PR
       │
       ▼
  quality   black --check + ruff check
       │
       ▼
  test      pytest tests/unit/ + pytest tests/integration/
       │  (push to main only)
       ▼
  build     docker buildx build → push to sameerhandsome12/cineagent-pro:latest
```

**Required GitHub Secrets:**

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | `sameerhandsome12` |
| `DOCKERHUB_TOKEN` | Docker Hub access token (Account Settings → Security) |

---

## Project Structure

```
cineagent-pro/
├── mcp_server/
│   ├── server.py                ← FastMCP entry point (port 8001)
│   └── tools/
│       ├── script_tools.py
│       ├── casting_tools.py
│       ├── budget_tools.py
│       └── market_tools.py
│
├── mcp_clients/
│   ├── loader.py                ← Loads local + remote MCP servers at startup
│   ├── tool_registry.py         ← Slices tool list per agent
│   └── remote_configs/
│       ├── brave_search.json
│       ├── playwright.json
│       ├── filesystem.json
│       └── github.json
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── agents/
│   │   ├── _base.py             ← Shared agentic loop
│   │   ├── orchestrator.py
│   │   ├── script_analyst.py
│   │   ├── budget_planner.py
│   │   ├── casting_director.py
│   │   └── market_intel.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   ├── prompt/
│   │   └── templates.py
│   ├── api/
│   │   ├── chat.py
│   │   ├── health.py
│   │   └── profile.py
│   ├── auth/
│   │   ├── router.py
│   │   └── jwt_handler.py
│   ├── database/
│   │   ├── connection.py
│   │   ├── models.py
│   │   ├── crud.py
│   │   └── schemas.py
│   ├── rag/
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   └── indexer.py
│   ├── cache/
│   │   └── redis_client.py
│   └── worker/
│       ├── celery_app.py
│       └── tasks.py
│
├── frontend/
│   ├── index.html
│   ├── chat.html
│   ├── profile.html
│   ├── css/
│   └── js/
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
│
├── k8s/
├── docker/
├── .github/workflows/ci-cd.yml
├── requirements.txt
├── pyproject.toml
└── pytest.ini
```