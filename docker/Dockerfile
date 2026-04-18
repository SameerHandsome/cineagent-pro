# ══════════════════════════════════════════════════════════════════════════════
# CineAgent Pro — Multi-stage Dockerfile
#
# Stage 1 (builder): install Python dependencies into a venv
# Stage 2 (runtime): copy only the venv + source — no build tools in prod image
#
# Result: ~400MB image vs ~1.2GB single-stage equivalent
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps needed only at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first — layer cache: only re-installs when requirements.txt changes
COPY requirements.txt .

# Create venv inside /build/venv so we can copy it cleanly to runtime stage
RUN python -m venv /build/venv \
    && /build/venv/bin/pip install --upgrade pip --no-cache-dir \
    && /build/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps only — no compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Node/npm for Playwright MCP and Filesystem MCP (stdio subprocess servers)
RUN npm install -g npx

# Non-root user for security
RUN useradd --create-home --shell /bin/bash cineagent
WORKDIR /app

# Copy venv from builder
COPY --from=builder /build/venv /app/venv

# Copy application source
COPY --chown=cineagent:cineagent . /app

# Create uploads directory
RUN mkdir -p /app/uploads && chown cineagent:cineagent /app/uploads

USER cineagent

# Activate venv for all subsequent commands
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app"

# Health check — calls /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Default: run the FastAPI backend
# Override CMD in k8s/docker-compose for the worker
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
