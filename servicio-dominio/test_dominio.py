"""Pruebas del microservicio de dominio (las ejecuta el workflow antes del build).

- Unitarias (sin base): contrato de entrada, PAN rechazado, política de reserva.
- Integración (con DATABASE_URL y la base cargada por servicio-carga): flujo /evaluar completo,
  con el microservicio de IA simulado (respondiendo, y caído) mediante httpx.MockTransport.
"""
import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from main import Transaccion, politica_reserva

BASE = {"usuario_id": "KP301626", "comercio_id": "CM40294", "monto": 25000, "moneda": "CLP", "canal": "app",
        "mcc": "5411", "pais_transaccion": "cl"}


def test_pan_completo_es_rechazado():
    with pytest.raises(ValueError):
        Transaccion(**BASE, tarjeta_enmascarada="4111 1111 1111 1111")
    assert Transaccion(**BASE, tarjeta_enmascarada="****1111").pais_transaccion == "CL"


def test_politica_de_reserva():
    t = Transaccion(**BASE)
    assert politica_reserva(t, 26, False)[0] == "aprobar"
    assert politica_reserva(t, 500, False)[0] == "revisar"
    assert politica_reserva(t, 26, True)[0] == "bloquear"       # la regla dura siempre manda


requiere_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="sin base de datos")


def ia_simulada(request: httpx.Request) -> httpx.Response:
    e = json.loads(request.content)
    assert e["usuario"] is not None                             # el dominio envía el perfil leído de la base
    return httpx.Response(200, json={
        "transaccion_id": e["transaccion_id"], "decision": "revisar", "probabilidad_fraude": 0.41,
        "razones": [{"factor": "log_monto_usd", "descripcion": "Aporte del factor log_monto_usd", "valor": 3.2,
                     "aporte_logodds": 0.8, "efecto": "sube el riesgo"}],
        "version_modelo": "test", "version_umbrales": "u-test", "latencia_ms": 1.0, "regla_aplicada": None})


def ia_caida(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("ia caída")


@requiere_db
def test_evaluar_guarda_la_prediccion(monkeypatch):
    monkeypatch.setattr(main, "http", httpx.Client(transport=httpx.MockTransport(ia_simulada)))
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200
        r = c.post("/evaluar", json=BASE)
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["decision"] == "revisar" and b["origen"] == "modelo" and b["version_modelo"] == "test"
        guardada = c.get(f"/predicciones/{b['transaccion_id']}").json()
        assert guardada[0]["version_modelo"] == "test"          # RF7: se puede reproducir


@requiere_db
def test_ia_caida_aplica_politica_de_reserva(monkeypatch):
    monkeypatch.setattr(main, "http", httpx.Client(transport=httpx.MockTransport(ia_caida)))
    with TestClient(main.app) as c:
        b = c.post("/evaluar", json=BASE).json()
        assert b["origen"] == "reserva" and b["decision"] in ("aprobar", "revisar", "bloquear")
