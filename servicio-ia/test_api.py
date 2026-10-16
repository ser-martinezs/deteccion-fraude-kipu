"""
Tests del microservicio de IA. Requieren servicio-ia/model/ (lo genera entrenar.py; en CI corre antes que pytest).

    python datos/preparar.py && python servicio-ia/entrenar.py && pytest servicio-ia
"""
import math

import pytest
from fastapi.testclient import TestClient

import main as main_module

UMBRALES = {"revisar_dia": 0.30, "bloquear_dia": 0.70, "bloquear_noche": 0.60}
TURNO = [8, 22]

ENTRADA_CONOCIDA = {
    "transaccion_id": "test_ci_001",
    "usuario_id": "usr_test",
    "comercio_id": "com_test",
    "timestamp": "2026-10-15T20:00:00Z",
    "monto": 10000,
    "canal": "app",
    "pais_transaccion": "CL",
    "mcc": 5411,
    "intentos_previos_1h": 0.0,
}

PERFIL = {
    "usuario": {"pais_usuario": "CL", "segmento": "persona", "verificado": 1, "telefono_verificado": 1,
                "nivel_kyc": 2, "dominio_desechable": 0, "fecha_registro": "2024-03-10"},
    "dispositivo": {"tipo_dispositivo": "android", "pais_disp": "CL", "emulador": 0},
}


@pytest.fixture(scope="module")
def client():
    with TestClient(main_module.app) as c:      # el `with` ejecuta el lifespan: carga el modelo
        yield c


def _post(client, **cambios):
    return client.post("/predict", json={**ENTRADA_CONOCIDA, **cambios})


# ------------------------------------------------------------------ salud y contrato
def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["version_modelo"]


def test_predict_entrada_conocida(client):
    r = _post(client)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["transaccion_id"] == "test_ci_001"
    assert d["decision"] in ("aprobar", "revisar", "bloquear")
    assert 0.0 <= d["probabilidad_fraude"] <= 1.0
    assert d["version_modelo"] and d["version_umbrales"]
    assert d["latencia_ms"] >= 0
    assert len(d["razones"]) == 3
    for razon in d["razones"]:
        assert razon["efecto"] in ("sube el riesgo", "baja el riesgo")
        assert math.isfinite(razon["aporte_logodds"])
    assert d["regla_aplicada"] is None


def test_predict_es_determinista(client):
    a, b = _post(client).json(), _post(client).json()
    assert a["probabilidad_fraude"] == b["probabilidad_fraude"]
    assert a["decision"] == b["decision"]


def test_predict_con_perfil_enriquecido(client):
    r = _post(client, **PERFIL)
    assert r.status_code == 200, r.text
    assert 0.0 <= r.json()["probabilidad_fraude"] <= 1.0


def test_respuesta_sin_perfil_no_rompe_por_nan(client):
    """Sin perfil casi todas las variables son NaN: la respuesta igual debe ser JSON válido (valor null)."""
    r = _post(client, transaccion_id="sin_perfil")
    assert r.status_code == 200, r.text
    for razon in r.json()["razones"]:
        assert razon["valor"] is None or isinstance(razon["valor"], (int, float, str, bool))


# ------------------------------------------------------------------ formatos que el entrenamiento sí vio
@pytest.mark.parametrize("monto", ["$1.200", "1.200", "45,50", "$47.33", 1200, 45.5, 1.265])
def test_formatos_de_monto(client, monto):
    assert _post(client, monto=monto).status_code == 200


@pytest.mark.parametrize("ts", ["2026-10-15T20:00:00Z", "2026-10-15T16:00:00-04:00",
                                "2026-10-15 16:00:00", "15/10/2026 16:00"])
def test_formatos_de_timestamp_equivalentes(client, ts):
    """Los cuatro formatos son el mismo instante (20:00 UTC = 16:00 en Chile): deben dar la misma probabilidad."""
    base = _post(client, timestamp="2026-10-15T20:00:00Z").json()["probabilidad_fraude"]
    r = _post(client, timestamp=ts)
    assert r.status_code == 200, r.text
    assert r.json()["probabilidad_fraude"] == base


def test_categorias_con_otra_escritura(client):
    a = _post(client, canal="app", pais_transaccion="CL").json()["probabilidad_fraude"]
    b = _post(client, canal="movil", pais_transaccion="Chile ").json()["probabilidad_fraude"]
    assert a == b


# ------------------------------------------------------------------ reglas de decisión
def test_regla_dura_de_lista_negra_bloquea(client):
    r = _post(client, lista_negra_dura=True)
    assert r.status_code == 200
    assert r.json()["decision"] == "bloquear"
    assert r.json()["regla_aplicada"]


@pytest.mark.parametrize("p,hora,esperado", [
    (0.10, 12, "aprobar"), (0.40, 12, "revisar"), (0.80, 12, "bloquear"),
    (0.40, 3, "aprobar"),      # de madrugada no hay zona 'revisar'
    (0.65, 3, "bloquear"),     # y se bloquea con el umbral nocturno
    (0.65, 12, "revisar"),     # el mismo p de día solo se revisa
    (0.80, 7.9, "bloquear"), (0.40, 22, "aprobar"),   # 22:00 ya es fuera de turno
])
def test_decidir_zonas(p, hora, esperado):
    assert main_module.decidir(p, hora, False, UMBRALES, TURNO) == esperado


def test_decidir_regla_dura_manda_sobre_el_modelo():
    assert main_module.decidir(0.0, 12, True, UMBRALES, TURNO) == "bloquear"


# ------------------------------------------------------------------ entradas inválidas -> 422 (nunca 500)
@pytest.mark.parametrize("cambios", [
    {"monto": "abc"}, {"monto": None}, {"monto": True},
    {"timestamp": "no-es-una-fecha"}, {"timestamp": "2026-10-15T20:00:00"},
    {"canal": "telepatia"}, {"pais_transaccion": "Narnia"},
    {"mcc": "abc"}, {"moneda": "EUR"},
    {"intentos_previos_1h": -1},
    {"usuario": {"nivel_kyc": 9}}, {"usuario": {"segmento": "otro"}},
    {"usuario": {"fecha_registro": "ayer"}}, {"dispositivo": {"tipo_dispositivo": "nokia"}},
])
def test_entrada_invalida_responde_422(client, cambios):
    r = _post(client, **cambios)
    assert r.status_code == 422, f"{cambios} -> {r.status_code} {r.text}"
    assert "Traceback" not in r.text


def test_faltan_campos_obligatorios(client):
    sin_canal = {k: v for k, v in ENTRADA_CONOCIDA.items() if k != "canal"}
    assert client.post("/predict", json=sin_canal).status_code == 422


# ------------------------------------------------------------------ modelo no disponible
def test_modelo_no_listo_responde_503(client, monkeypatch):
    monkeypatch.setitem(main_module.estado, "listo", False)
    assert client.get("/health").status_code == 503
    assert _post(client).status_code == 503


def test_error_interno_no_filtra_detalles(client, monkeypatch):
    def roto(*a, **k):
        raise RuntimeError("secreto interno /app/model/x")
    monkeypatch.setattr(main_module, "construir_features", roto)
    r = _post(client)
    assert r.status_code == 500
    assert "secreto" not in r.text and "/app" not in r.text
