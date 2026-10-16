from fastapi.testclient import TestClient
from servicio-ia.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_transaccion_valida():
    payload = {
        "transaccion_id": "test_ci_001",
        "usuario_id": "usr_test",
        "comercio_id": "com_test",
        "timestamp": "2026-10-15T20:00:00Z",
        "monto": 10000,
        "canal": "app",
        "pais_transaccion": "CL",
        "mcc": 5411,
        "intentos_previos_1h": 0.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert data["decision"] in ["aprobar", "revisar", "bloquear"]
    assert "version_modelo" in data
