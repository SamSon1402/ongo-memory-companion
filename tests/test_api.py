"""FastAPI integration tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ongomemory.api import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200


def test_write_and_list_episodes(client: TestClient) -> None:
    r = client.post(
        "/api/v1/users/u1/episodes",
        json={"text": "call with Joonatan", "topic": "calendar"},
    )
    assert r.status_code == 200
    assert r.json()["text"] == "call with Joonatan"

    r2 = client.get("/api/v1/users/u1/episodes")
    assert r2.status_code == 200
    assert len(r2.json()) == 1


def test_recall(client: TestClient) -> None:
    client.post(
        "/api/v1/users/u2/episodes",
        json={"text": "I had coffee with Joonatan this morning", "topic": "social"},
    )
    client.post(
        "/api/v1/users/u2/episodes",
        json={"text": "I went to the gym", "topic": "health"},
    )
    r = client.post(
        "/api/v1/users/u2/recall",
        json={"text": "did I see Joonatan?", "k": 3},
    )
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) >= 1
    assert "joonatan" in hits[0]["text"].lower()
    # Decomposed scores are reported.
    for hit in hits:
        assert "vec_similarity" in hit
        assert "recency_score" in hit
        assert "entity_overlap" in hit


def test_facts(client: TestClient) -> None:
    client.post(
        "/api/v1/users/u3/episodes",
        json={"text": "Hey, I'm Sam.", "topic": "profile"},
    )
    r = client.get("/api/v1/users/u3/facts")
    assert r.status_code == 200
    assert r.json().get("name") == "Sam"


def test_identity_endpoint(client: TestClient) -> None:
    r = client.post(
        "/api/v1/users/u4/identity",
        json={"face_id": "f7a2", "display_name": "Sam"},
    )
    assert r.status_code == 204


def test_validation_rejects_empty_text(client: TestClient) -> None:
    r = client.post("/api/v1/users/u1/episodes", json={"text": ""})
    assert r.status_code == 422


def test_validation_rejects_oversized_k(client: TestClient) -> None:
    r = client.post(
        "/api/v1/users/u1/recall", json={"text": "x", "k": 9999}
    )
    assert r.status_code == 422
