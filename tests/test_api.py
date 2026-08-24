import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_session_lifecycle(client):
    created = client.post("/sessions", json={"title": "q4 review"})
    assert created.status_code == 201
    session_id = created.json()["id"]

    listed = client.get("/sessions").json()
    assert any(s["id"] == session_id for s in listed)

    assert client.get(f"/sessions/{session_id}/messages").json() == []
    assert client.get(f"/sessions/{session_id}/reports").json() == []


def test_unknown_session_is_404(client):
    assert client.get("/sessions/999999/messages").status_code == 404
    assert client.post("/chat", json={"session_id": 999999, "message": "hi"}).status_code == 404


def test_research_rejects_short_question(client):
    assert client.post("/research", json={"question": "hm"}).status_code == 422
