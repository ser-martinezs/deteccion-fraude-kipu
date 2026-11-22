"""Pruebas de EP3 del dominio y del worker del LLM (las ejecuta el workflow `dominio` antes de publicar la imagen).

Obligatorias de la sección 4.3 del enunciado:
  1. POST /api/predict válido -> 200 con resultado y versión de modelo
  2. Campo faltante o de tipo incorrecto -> 422 con el detalle
  3. Sin X-API-Key -> 401
  5. RUT, correo, teléfono y tarjeta pasados por la anonimización: ninguno sobrevive
  6. LLM con 429 o caído -> la predicción responde igual y el estado queda en modo degradado
  4. Ráfaga sobre el límite de solicitudes -> 429

La IA (SageMaker) y NVIDIA se simulan: el pipeline no necesita AWS ni la clave de NVIDIA.
"""
import os

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import main
import worker
from anonimizar import CAMPOS_PERMITIDOS, anonimizar, mensaje_para_llm

CLAVE = "clave-de-prueba"
H = {"X-API-Key": CLAVE}
BASE = {"usuario_id": "KP301626", "comercio_id": "CM40294", "monto": 25000, "moneda": "CLP", "canal": "app",
        "mcc": "5411", "pais_transaccion": "CL"}
requiere_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="sin base de datos")


@pytest.fixture(autouse=True)
def clave(monkeypatch):
    monkeypatch.setattr(main, "CLAVES_API", [CLAVE])
    monkeypatch.setattr(main, "limitador", main.LimitadorPorMinuto())    # cada prueba parte con el contador en cero


def ia_que_decide(decision, prob=0.41):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "transaccion_id": "x", "decision": decision, "probabilidad_fraude": prob,
            "razones": [{"factor": "log_monto_usd", "descripcion": "Monto alto para el usuario", "valor": 3.2,
                         "aporte_logodds": 0.8, "efecto": "sube el riesgo"}],
            "version_modelo": "3.0.0-test", "version_umbrales": "u-test", "latencia_ms": 1.0, "regla_aplicada": None})
    return httpx.Client(transport=httpx.MockTransport(responder))


def error_openai(clase, status):
    resp = httpx.Response(status, request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"))
    return clase("simulado", response=resp, body=None)


class LLMFalso:
    """Imita OpenAI().chat.completions.create: lanza los errores de la lista y luego responde."""
    def __init__(self, errores=(), texto="Resumen de prueba"):
        self.errores, self.texto, self.llamadas = list(errores), texto, 0
        self.chat = self
        self.completions = self

    def create(self, **_):
        self.llamadas += 1
        if self.errores:
            raise self.errores.pop(0)
        msg = type("M", (), {"content": self.texto})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


# ------------------------------------------------------------------ 3 · sin clave -> 401
def test_3_sin_clave_responde_401():
    with TestClient(main.app) as c:
        assert c.post("/predict", json=BASE).status_code == 401
        assert c.post("/predict", json=BASE, headers={"X-API-Key": "otra"}).status_code == 401
        assert c.get("/revision").status_code == 401


# ------------------------------------------------------------------ 2 · entrada inválida -> 422 con detalle
@pytest.mark.parametrize("cambio", [{"monto": "abc"}, {"usuario_id": None}, {"canal": "fax"}, {"mcc": "54"}])
def test_2_entrada_invalida_responde_422_con_detalle(cambio):
    cuerpo = {k: v for k, v in {**BASE, **cambio}.items() if v is not None}
    with TestClient(main.app) as c:
        r = c.post("/predict", json=cuerpo, headers=H)
    assert r.status_code == 422
    detalle = r.json()["detail"]
    assert isinstance(detalle, list) and detalle[0]["loc"] and detalle[0]["msg"]


def test_campo_tarjeta_con_numero_completo_responde_422():
    """RNF5: un campo fuera del contrato (aquí el número de tarjeta) no se ignora en silencio: 422."""
    with TestClient(main.app) as c:
        r = c.post("/predict", json={**BASE, "tarjeta": "4111111111111111"}, headers=H)
    assert r.status_code == 422 and "tarjeta" in r.text


# ------------------------------------------------------------------ 4 · ráfaga sobre el límite -> 429
def test_4_rafaga_sobre_el_limite_responde_429(monkeypatch):
    monkeypatch.setattr(main, "LIMITE_POR_MINUTO", 5)
    with TestClient(main.app) as c:
        # cuerpo inválido a propósito: el límite se evalúa antes y la prueba no necesita base ni IA
        codigos = [c.post("/predict", json={}, headers=H).status_code for _ in range(8)]
        otro_cliente = c.post("/predict", json={}, headers={**H, "X-Forwarded-For": "200.1.2.3"}).status_code
    assert codigos[:5] == [422] * 5 and codigos[5:] == [429] * 3
    assert otro_cliente == 422                                   # el límite es por cliente, no global


def test_limitador_libera_despues_de_un_minuto():
    lim = main.LimitadorPorMinuto()
    assert all(lim.permitir("a", 2, ahora=t) for t in (0, 1))
    assert not lim.permitir("a", 2, ahora=30)
    assert lim.permitir("a", 2, ahora=61)


def test_pan_completo_responde_422():
    with TestClient(main.app) as c:
        r = c.post("/predict", json={**BASE, "tarjeta_enmascarada": "4111 1111 1111 1111"}, headers=H)
    assert r.status_code == 422 and "PAN" in r.text


# ------------------------------------------------------------------ 5 · anonimización
def test_5_anonimizacion_no_deja_pasar_datos_personales():
    texto = ("Cliente 12.345.678-5 (otro: 9876543-K) escribió desde juan.perez@correo.cl, "
             "teléfono +56 9 8765 4321, tarjeta 4111 1111 1111 1111")
    limpio = anonimizar(texto)
    for dato in ("12.345.678-5", "9876543-K", "juan.perez@correo.cl", "8765 4321", "4111 1111 1111 1111", "4111"):
        assert dato not in limpio, dato
    assert all(e in limpio for e in ("[RUT]", "[CORREO]", "[TELEFONO]", "[TARJETA]"))


def test_mensaje_para_llm_usa_lista_blanca():
    pred = {"prediccion_id": 7, "decision": "revisar", "probabilidad_fraude": 0.4, "monto_usd": 300,
            "usuario_id": "KP1", "nombre": "Juan Pérez", "rut": "12.345.678-5", "ip": "1.2.3.4",
            "tarjeta_enmascarada": "****1111", "razones": [{"factor": "x", "descripcion": "correo a@b.cl", "valor": "secreto"}]}
    msg = mensaje_para_llm(pred)
    assert set(msg) <= set(CAMPOS_PERMITIDOS)
    assert "Juan" not in str(msg) and "KP1" not in str(msg) and "1.2.3.4" not in str(msg) and "a@b.cl" not in str(msg)
    assert "valor" not in msg["razones"][0]


# ------------------------------------------------------------------ 6 · LLM con 429 o caído -> modo degradado
def test_6_llm_con_429_persistente_queda_degradado(monkeypatch):
    falso = LLMFalso([error_openai(openai.RateLimitError, 429)] * 10)
    monkeypatch.setattr(worker, "cliente", lambda: falso)
    esperas = []
    estado, resumen, detalle = worker.resumir({"prediccion_id": 1, "decision": "revisar"}, dormir=esperas.append)
    assert (estado, resumen) == ("degradado", None) and "429" in detalle
    assert falso.llamadas == worker.NVIDIA_REINTENTOS
    assert esperas == sorted(esperas) and esperas[0] < esperas[-1]          # espera creciente


def test_6_llm_caido_y_luego_responde(monkeypatch):
    req = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    falso = LLMFalso([openai.APIConnectionError(request=req), error_openai(openai.RateLimitError, 429)])
    monkeypatch.setattr(worker, "cliente", lambda: falso)
    estado, resumen, _ = worker.resumir({"prediccion_id": 1}, dormir=lambda s: None)
    assert estado == "listo" and resumen == "Resumen de prueba" and falso.llamadas == 3


def test_llm_con_clave_invalida_no_reintenta(monkeypatch):
    falso = LLMFalso([error_openai(openai.AuthenticationError, 401)] * 5)
    monkeypatch.setattr(worker, "cliente", lambda: falso)
    estado, _, detalle = worker.resumir({"prediccion_id": 1}, dormir=lambda s: None)
    assert estado == "degradado" and falso.llamadas == 1 and "clave" in detalle


@requiere_db
def test_6_cola_caida_la_prediccion_responde_igual_en_modo_degradado(monkeypatch):
    monkeypatch.setattr(main, "http", ia_que_decide("revisar"))
    def cola_caida(_):
        raise ConnectionError("rabbitmq caído")
    monkeypatch.setattr(main, "publicar_en_cola", cola_caida)
    with TestClient(main.app) as c:
        r = c.post("/predict", json=BASE, headers=H)
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["decision"] == "revisar" and b["estado_llm"] == "degradado" and b["version_modelo"] == "3.0.0-test"
        assert "ConnectionError" in b["motivo_llm"]
        g = c.get(f"/predicciones/{b['id']}", headers=H).json()
        assert g["estado_llm"] == "degradado" and "cola" in g["motivo_llm"]


# ------------------------------------------------------------------ 1 · predicción válida -> 200 con versión
@requiere_db
def test_1_predict_valido_responde_200_con_version_y_encola(monkeypatch):
    monkeypatch.setattr(main, "http", ia_que_decide("bloquear", 0.93))
    encolados = []
    monkeypatch.setattr(main, "publicar_en_cola", encolados.append)
    with TestClient(main.app) as c:
        r = c.post("/predict", json={**BASE, "tarjeta_enmascarada": "****1111", "ip": "10.0.0.1"}, headers=H)
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["decision"] == "bloquear" and b["version_modelo"] == "3.0.0-test" and b["razones"]
        assert b["estado_llm"] == "pendiente" and len(encolados) == 1
        assert "usuario_id" not in encolados[0] and "****1111" not in str(encolados[0]) and "10.0.0.1" not in str(encolados[0])
        guardada = c.get(f"/predicciones/{b['prediccion_id']}", headers=H).json()
        assert guardada["version_modelo"] == "3.0.0-test" and guardada["estado_llm"] == "pendiente"


@requiere_db
def test_aprobar_no_se_encola(monkeypatch):
    monkeypatch.setattr(main, "http", ia_que_decide("aprobar", 0.01))
    encolados = []
    monkeypatch.setattr(main, "publicar_en_cola", encolados.append)
    with TestClient(main.app) as c:
        b = c.post("/predict", json=BASE, headers=H).json()
    assert b["estado_llm"] == "no_aplica" and encolados == []


@requiere_db
def test_worker_guarda_el_resumen_y_el_estado_del_llm(monkeypatch):
    import psycopg
    monkeypatch.setattr(main, "http", ia_que_decide("revisar"))
    monkeypatch.setattr(main, "publicar_en_cola", lambda m: None)
    with TestClient(main.app) as c:
        pid = c.post("/predict", json=BASE, headers=H).json()["prediccion_id"]
        monkeypatch.setattr(worker, "cliente", lambda: LLMFalso(texto="Monto inusual: revisar historial"))
        fallidos = []
        with psycopg.connect(os.environ["DATABASE_URL"]) as con:
            assert worker.procesar(con, f'{{"prediccion_id": {pid}}}'.encode(), fallidos.append) == "listo"
        g = c.get(f"/predicciones/{pid}", headers=H).json()
        assert g["estado_llm"] == "listo" and g["resumen_llm"] == "Monto inusual: revisar historial" and fallidos == []
        assert c.get("/health").json()["piezas"]["llm"]["estado"] == "ok"


@requiere_db
def test_6_worker_con_clave_invalida_deja_motivo_y_mensaje_en_fallidos(monkeypatch):
    import psycopg
    monkeypatch.setattr(main, "http", ia_que_decide("bloquear", 0.97))
    monkeypatch.setattr(main, "publicar_en_cola", lambda m: None)
    with TestClient(main.app) as c:
        pid = c.post("/predict", json=BASE, headers=H).json()["id"]
        monkeypatch.setattr(worker, "cliente", lambda: LLMFalso([error_openai(openai.AuthenticationError, 401)]))
        fallidos = []
        with psycopg.connect(os.environ["DATABASE_URL"]) as con:
            assert worker.procesar(con, f'{{"prediccion_id": {pid}}}'.encode(), fallidos.append) == "degradado"
        g = c.get(f"/predicciones/{pid}", headers=H).json()
        assert g["estado_llm"] == "degradado" and "401" in g["motivo_llm"] and g["decision"] == "bloquear"
        assert len(fallidos) == 1
        assert c.get("/health").json()["piezas"]["llm"]["estado"] == "degradado"


@requiere_db
def test_revision_ordenada_por_riesgo_y_monto(monkeypatch):
    monkeypatch.setattr(main, "publicar_en_cola", lambda m: None)
    with TestClient(main.app) as c:
        for prob, monto in [(0.30, 9000), (0.95, 1000), (0.95, 500000)]:
            monkeypatch.setattr(main, "http", ia_que_decide("revisar", prob))
            assert c.post("/predict", json={**BASE, "monto": monto}, headers=H).status_code == 200
        filas = c.get("/revision", headers=H).json()
    perdidas = [float(f["perdida_esperada_usd"]) for f in filas if f["perdida_esperada_usd"] is not None]
    assert perdidas == sorted(perdidas, reverse=True) and len(perdidas) >= 3


@requiere_db
def test_ia_que_rechaza_la_entrada_devuelve_422(monkeypatch):
    def ia_422(request):
        return httpx.Response(422, json={"detail": "mcc: debe ser un código numérico"})
    monkeypatch.setattr(main, "http", httpx.Client(transport=httpx.MockTransport(ia_422)))
    with TestClient(main.app) as c:
        r = c.post("/predict", json=BASE, headers=H)
    assert r.status_code == 422 and "mcc" in r.text


@requiere_db
def test_health_es_publico_y_reporta_cada_pieza():
    with TestClient(main.app) as c:
        r = c.get("/health")
        assert c.get("/vivo").status_code == 200
    assert r.status_code == 200
    assert set(r.json()["piezas"]) == {"base", "sagemaker", "cola", "llm"}
