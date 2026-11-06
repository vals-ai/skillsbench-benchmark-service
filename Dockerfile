FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY image-manifest.json ./
COPY src ./src
COPY skillsbench ./skillsbench

ENV SKILLSBENCH_REPO_ROOT=/app/skillsbench
ENV SKILLSBENCH_VALKYRIE_IMAGE_MANIFEST=/app/image-manifest.json
RUN uv sync --locked --no-dev

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "skillsbench_valkyrie.main:app", "--host", "0.0.0.0", "--port", "8001", "--ws-ping-interval", "30", "--ws-ping-timeout", "10"]
