# TiO2Agent API image. Usually built and started by Compose: docker compose up -d --build
# Standalone:
#   docker build -t tio2agent .
#   docker run -d -p 8000:8000 -v tio2-data:/app/data tio2agent
# Settings are TIO2_* environment variables (see .env.example). Do not pass the local
# .env with --env-file: its localhost URLs point at the container itself.

# ---- build: resolve dependencies from uv.lock into /app/.venv -------------------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so source changes don't invalidate this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY pyproject.toml uv.lock README.md LICENSE* ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- runtime -----------------------------------------------------------------------
FROM python:3.11-slim

LABEL org.opencontainers.image.title="TiO2Agent" \
      org.opencontainers.image.description="RAG + multi-step Agent QA backend for TiO2 technical documents"

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown app:app /app/data

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TIO2_HOST=0.0.0.0 \
    TIO2_PORT=8000 \
    TIO2_DATA_DIR=/app/data

WORKDIR /app
USER app
EXPOSE 8000
VOLUME ["/app/data"]

# Reads TIO2_PORT, so the check still works if the listening port is changed.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request; port=os.environ.get('TIO2_PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4).status == 200 else 1)"]

CMD ["tio2agent"]
