import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client(desk):
    return TestClient(create_app(desk))


def _create(client) -> dict:
    response = client.post(
        "/tickets",
        json={
            "subject": "Forgot my password",
            "body": "I forgot my password and can't log in to my account.",
        },
    )
    assert response.status_code == 201
    return response.json()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_ticket_reaches_approval_gate(client):
    ticket = _create(client)
    assert ticket["status"] == "awaiting_approval"
    assert ticket["category"] == "account_access"
    assert ticket["draft_response"]
    assert "password-reset.md" in ticket["citations"]


def test_get_ticket(client):
    ticket = _create(client)
    response = client.get(f"/tickets/{ticket['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == ticket["id"]


def test_get_unknown_ticket_404(client):
    assert client.get("/tickets/nope").status_code == 404


def test_approve_endpoint_finalises(client):
    ticket = _create(client)
    response = client.post(f"/tickets/{ticket['id']}/approve")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "finalised"
    assert body["final_response"] == body["draft_response"]


def test_reject_endpoint(client):
    ticket = _create(client)
    response = client.post(
        f"/tickets/{ticket['id']}/reject", json={"reason": "escalate to human"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["rejection_reason"] == "escalate to human"


def test_approve_unknown_ticket_404(client):
    assert client.post("/tickets/nope/approve").status_code == 404
    assert client.post("/tickets/nope/reject", json={}).status_code == 404
