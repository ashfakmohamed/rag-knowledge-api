from fastapi.testclient import TestClient

from app.main import app


def test_health_reports_loaded_documents():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["documents"] >= 2


def test_query_returns_grounded_citation():
    with TestClient(app) as client:
        response = client.post(
            "/query", json={"question": "How is the portfolio deployed?", "top_k": 2}
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["grounded"] is True
    assert payload["citations"][0]["source"].endswith("portfolio.md")


def test_document_can_be_ingested_and_retrieved():
    document = {
        "title": "Voice AI",
        "content": "LiveKit and Twilio support real-time enterprise voice applications.",
        "source": "test://voice-ai",
    }
    with TestClient(app) as client:
        create_response = client.post("/documents", json=document)
        query_response = client.post(
            "/query", json={"question": "Which tools support voice applications?"}
        )

    assert create_response.status_code == 201
    assert query_response.status_code == 200
    assert any(
        citation["source"] == "test://voice-ai"
        for citation in query_response.json()["citations"]
    )


def test_query_validation_rejects_invalid_top_k():
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "valid question", "top_k": 0})

    assert response.status_code == 422
