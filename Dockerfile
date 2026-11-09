# One image for the assaygate service and the demo classifier (different commands).
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY examples ./examples
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"
RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app/data
USER app
EXPOSE 8100
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health').status==200 else 1)"
CMD ["assaygate", "serve", "--host", "0.0.0.0", "--port", "8100", "--runs-dir", "/app/runs"]
