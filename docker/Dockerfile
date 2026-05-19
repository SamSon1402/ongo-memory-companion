# OngoMemory-Companion edge image.
# Runs the memory FastAPI on the device. SQLite store on a mounted volume.
#
#   docker run -p 8002:8002 -v /var/lib/ongo:/data ongomemory/edge:latest

FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

# Persistent store mount point — SQLite goes here.
VOLUME ["/data"]
ENV ONGO_MEMORY_DB=/data/memory.db

EXPOSE 8002

HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8002/healthz || exit 1

CMD ["python", "-m", "uvicorn", "ongomemory.api:app", "--host", "0.0.0.0", "--port", "8002"]
