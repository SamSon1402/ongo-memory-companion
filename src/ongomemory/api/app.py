"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ongomemory.api.routes import health, memory


def create_app() -> FastAPI:
    app = FastAPI(
        title="OngoMemory-Companion API",
        version="0.1.0",
        description=(
            "Living memory layer for the Ongo companion robot. "
            "Per-user episodes, facts, habits, and recall."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:8002"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.include_router(health.router, tags=["health"])
    app.include_router(memory.router, prefix="/api/v1", tags=["memory"])
    return app


app = create_app()
