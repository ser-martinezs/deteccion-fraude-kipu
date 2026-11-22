"""
Tests del microservicio de IA (EP3). NO necesitan AWS ni modelo: el cliente de SageMaker (boto3) se reemplaza por uno
simulado con unittest.mock.patch. El simulado responde con el contrato real del endpoint (aws/src/inference.py):
    {"probabilidad": ..., "aportes": {factor: log-odds}, "version_modelo": ...}
y de forma DETERMINISTA según las variables recibidas, de modo que "misma entrada -> misma salida" prueba que el
servicio arma siempre las mismas variables.

    python datos/preparar.py && pytest servicio-ia        (preparar.py solo hace falta para test_paridad.py)
"""
import hashlib
import io
import json
import math
from unittest import mock

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError, NoCredentialsError, ReadTimeoutError
from fastapi.testclient import TestClient

import main as main_module

UMBRALES = {"revisar_dia": 0.30, "bloquear_dia": 0.70, "bloquear_noche": 0.60}
TURNO = [8, 22]
FACTORES = list(main_module.preparar.FEATURES_NUM) + list(main_module.preparar.FEATURES_CAT)
VERSION_FALSA = "3.0.0-test"

ENTRADA_CONOCIDA = {
    "transaccion_id": "test_ci_001",
    "usuario_id": "usr_test",
    "comercio_id": "com_test",
    "timestamp": "2026-10-15T20:00:00Z",          # 17:00 en Chile: dentro del turno de analistas
    "monto": 10000,
    "canal": "app",
    "pais_transaccion": "CL",
    "mcc": 5411,
    "intentos_previos_1h": 0.0,
}
DE_NOCHE = "2026-10-15T06:00:00Z"                  # 03:00 en Chile: fuera de turno

PERFIL = {
    "usuario": {"pais_usuario": "CL", "segmento": "persona", "verificado": 1, "telefono_verificado": 1,
                "nivel_kyc": 2, "dominio_desechable": 0, "fecha_registro": "2024-03-10"},
    "dispositivo": {"tipo_dispositivo": "android", "pais_disp": "CL", "emulador": 0},
}


class SageMakerFalso:
    """Hace de cliente 'sagemaker-runtime' y 'sagemaker'. Registra cada llamada para poder inspeccionarla."""

    def __init__(self):
        self.llamadas = []              # cuerpos enviados a invoke_endpoint
        self.metadatos = []             # (EndpointName, ContentType, Accept) de cada invocación
        self.config_clientes = {}       # kwargs con que el servicio creó cada cliente de boto3
        self.p_forzada = None           # probabilidad fija (para probar umbrales)
        self.falla = None               # excepción que lanza invoke_endpoint
        self.cuerpo_crudo = None        # bytes que devuelve invoke_endpoint (para probar respuestas fuera de contrato)
        self.ultima = None              # última respuesta válida entregada
        self.estado = "InService"
        self.describe_falla = None
        self.describe_llamadas = 0

    def crear_cliente(self, servicio, **kwargs):          # reemplaza a boto3.client
        self.config_clientes[servicio] = kwargs
        return self

    def invoke_endpoint(self, EndpointName, ContentType, Accept, Body):
        cuerpo = json.loads(Body)
        self.llamadas.append(cuerpo)
        self.metadatos.append((EndpointName, ContentType, Accept))
        if self.falla is not None:
            raise self.falla
        if self.cuerpo_crudo is not None:
            return {"Body": io.BytesIO(self.cuerpo_crudo)}
        h = hashlib.sha256(json.dumps(cuerpo["variables"], sort_keys=True).encode()).digest()
        p = self.p_forzada if self.p_forzada is not None else (int.from_bytes(h[:4], "big") / 2 ** 32) * 0.9
        aportes = {f: round(((h[4 + i % 20] / 255) * 2 - 1) * 2, 4) for i, f in enumerate(FACTORES)}
        self.ultima = {"probabilidad": p, "aportes": aportes, "version_modelo": VERSION_FALSA}
        return {"Body": io.BytesIO(json.dumps(self.ultima).encode())}

    def describe_endpoint(self, EndpointName):
        self.describe_llamadas += 1
        if self.describe_falla is not None:
            raise self.describe_falla
        return {"EndpointStatus": self.estado}


@pytest.fixture
def falso():
    f = SageMakerFalso()
    with mock.patch.object(main_module.boto3, "client", side_effect=f.crear_cliente):
        yield f


@pytest.fixture
def client(falso):
    with TestClient(main_module.app) as c:      # el `with` ejecuta el lifespan: carga umbrales y crea los clientes (simulados)
        yield c


def _post(client, **cambios):
    return client.post("/predict", json={**ENTRADA_CONOCIDA, **cambios})


def _umbrales():
    return main_module.estado["umbrales"]["umbrales"]


# ------------------------------------------------------------------ contrato con el dominio
def test_predict_entrada_conocida(client, falso):
    r = _post(client)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["transaccion_id"] == "test_ci_001"
    assert d["decision"] in ("aprobar", "revisar", "bloquear")
    assert 0.0 <= d["probabilidad_fraude"] <= 1.0
    assert d["version_modelo"] == VERSION_FALSA and d["version_umbrales"]
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


def test_predict_con_perfil_enriquecido(client, falso):
    assert _post(client, **PERFIL).status_code == 200
    sin_perfil = _post(client, transaccion_id="otra").status_code
    assert sin_perfil == 200
    con, sin = falso.llamadas[0]["variables"], falso.llamadas[1]["variables"]
    assert sum(v is not None for v in con.values()) > sum(v is not None for v in sin.values())


# ------------------------------------------------------------------ lo que se envía a SageMaker
def test_envia_al_endpoint_las_variables_del_modelo(client, falso):
    assert _post(client, **PERFIL).status_code == 200
    assert len(falso.llamadas) == 1
    assert falso.metadatos[0] == (main_module.ENDPOINT, "application/json", "application/json")
    variables = falso.llamadas[0]["variables"]
    assert list(variables) == FACTORES                                    # las 21 variables, en el orden de preparar.py
    for c in main_module.preparar.FEATURES_NUM:
        assert variables[c] is None or isinstance(variables[c], (int, float)), c
    for c in main_module.preparar.FEATURES_CAT:
        assert variables[c] is None or isinstance(variables[c], str), c


def test_sin_perfil_envia_nulos_y_la_respuesta_es_json_valido(client, falso):
    """Sin perfil casi todas las variables son faltantes: viajan como null (nunca NaN) y la respuesta sigue siendo válida."""
    r = _post(client, transaccion_id="sin_perfil")
    assert r.status_code == 200, r.text
    assert any(v is None for v in falso.llamadas[0]["variables"].values())
    for razon in r.json()["razones"]:
        assert razon["valor"] is None or isinstance(razon["valor"], (int, float, str, bool))


# ------------------------------------------------------------------ razones
def test_razones_son_los_tres_mayores_aportes_en_valor_absoluto(client, falso):
    d = _post(client).json()
    esperado = sorted(falso.ultima["aportes"].items(), key=lambda kv: -abs(kv[1]))[:3]
    assert [x["factor"] for x in d["razones"]] == [f for f, _ in esperado]
    for razon, (_, aporte) in zip(d["razones"], esperado):
        assert razon["aporte_logodds"] == round(aporte, 3)
        assert razon["efecto"] == ("sube el riesgo" if aporte > 0 else "baja el riesgo")


# ------------------------------------------------------------------ formatos que el entrenamiento sí vio
@pytest.mark.parametrize("monto", ["$1.200", "1.200", "45,50", "$47.33", 1200, 45.5, 1.265])
def test_formatos_de_monto(client, monto):
    assert _post(client, monto=monto).status_code == 200


@pytest.mark.parametrize("ts", ["2026-10-15T20:00:00Z", "2026-10-15T16:00:00-04:00",
                                "2026-10-15 16:00:00", "15/10/2026 16:00"])
def test_formatos_de_timestamp_equivalentes(client, ts):
    """Los cuatro formatos son el mismo instante (20:00 UTC = 16:00 en Chile): deben llegar al modelo con las mismas variables."""
    base = _post(client, timestamp="2026-10-15T20:00:00Z").json()["probabilidad_fraude"]
    r = _post(client, timestamp=ts)
    assert r.status_code == 200, r.text
    assert r.json()["probabilidad_fraude"] == base


def test_categorias_con_otra_escritura(client):
    a = _post(client, canal="app", pais_transaccion="CL").json()["probabilidad_fraude"]
    b = _post(client, canal="movil", pais_transaccion="Chile ").json()["probabilidad_fraude"]
    assert a == b


# ------------------------------------------------------------------ reglas de decisión
def test_regla_dura_de_lista_negra_bloquea_aunque_el_modelo_diga_cero(client, falso):
    falso.p_forzada = 0.0
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


def test_la_decision_usa_los_umbrales_de_configuracion(client, falso):
    u = _umbrales()
    assert u["revisar_dia"] < u["bloquear_dia"]
    assert 0.0 < u["bloquear_noche"] <= 1.0
    casos = [(ENTRADA_CONOCIDA["timestamp"], u["revisar_dia"] / 2, "aprobar"),
             (ENTRADA_CONOCIDA["timestamp"], (u["revisar_dia"] + u["bloquear_dia"]) / 2, "revisar"),
             (ENTRADA_CONOCIDA["timestamp"], min(u["bloquear_dia"] + 0.01, 1.0), "bloquear"),
             (DE_NOCHE, u["bloquear_noche"] / 2, "aprobar"),
             (DE_NOCHE, min(u["bloquear_noche"] + 0.01, 1.0), "bloquear")]
    for ts, p, esperado in casos:
        falso.p_forzada = p
        assert _post(client, timestamp=ts).json()["decision"] == esperado, (ts, p)


def test_umbrales_se_leen_del_archivo_configurado(falso, tmp_path, monkeypatch):
    ruta = tmp_path / "umbrales.json"
    ruta.write_text(json.dumps({"version": "u-test", "umbrales": {"revisar_dia": 0.2, "bloquear_dia": 0.5, "bloquear_noche": 0.4}}))
    monkeypatch.setattr(main_module, "RUTA_UMBRALES", ruta)
    falso.p_forzada = 0.3
    with TestClient(main_module.app) as c:
        d = _post(c).json()
    assert d["version_umbrales"] == "u-test" and d["decision"] == "revisar"


def test_umbrales_incoherentes_dejan_el_servicio_no_listo(falso, tmp_path, monkeypatch):
    ruta = tmp_path / "umbrales.json"
    ruta.write_text(json.dumps({"version": "malo", "umbrales": {"revisar_dia": 0.9, "bloquear_dia": 0.1, "bloquear_noche": 0.4}}))
    monkeypatch.setattr(main_module, "RUTA_UMBRALES", ruta)
    with TestClient(main_module.app) as c:
        assert c.get("/health").status_code == 503
        assert _post(c).status_code == 503
    assert falso.llamadas == []


# ------------------------------------------------------------------ entradas inválidas -> 422 (nunca 500, nunca llega a SageMaker)
@pytest.mark.parametrize("cambios", [
    {"monto": "abc"}, {"monto": None}, {"monto": True},
    {"timestamp": "no-es-una-fecha"}, {"timestamp": "2026-10-15T20:00:00"},
    {"canal": "telepatia"}, {"pais_transaccion": "Narnia"},
    {"mcc": "abc"}, {"moneda": "EUR"},
    {"intentos_previos_1h": -1},
    {"usuario": {"nivel_kyc": 9}}, {"usuario": {"segmento": "otro"}},
    {"usuario": {"fecha_registro": "ayer"}}, {"dispositivo": {"tipo_dispositivo": "nokia"}},
])
def test_entrada_invalida_responde_422(client, falso, cambios):
    r = _post(client, **cambios)
    assert r.status_code == 422, f"{cambios} -> {r.status_code} {r.text}"
    assert "Traceback" not in r.text
    assert falso.llamadas == []


def test_faltan_campos_obligatorios(client, falso):
    sin_canal = {k: v for k, v in ENTRADA_CONOCIDA.items() if k != "canal"}
    assert client.post("/predict", json=sin_canal).status_code == 422
    assert falso.llamadas == []


# ------------------------------------------------------------------ SageMaker no disponible -> 503 (el dominio aplica su política de reserva)
CAIDAS = [
    pytest.param(ReadTimeoutError(endpoint_url="https://runtime.sagemaker.us-east-1.amazonaws.com/secreto"), id="timeout-lectura"),
    pytest.param(ConnectTimeoutError(endpoint_url="https://runtime.sagemaker.us-east-1.amazonaws.com/secreto"), id="timeout-conexion"),
    pytest.param(EndpointConnectionError(endpoint_url="https://runtime.sagemaker.us-east-1.amazonaws.com/secreto"), id="sin-conexion"),
    pytest.param(NoCredentialsError(), id="sin-credenciales"),
    pytest.param(ClientError({"Error": {"Code": "ModelError", "Message": "secreto /opt/ml/model"}}, "InvokeEndpoint"), id="ModelError"),
    pytest.param(ClientError({"Error": {"Code": "ThrottlingException", "Message": "secreto"}}, "InvokeEndpoint"), id="throttling"),
    pytest.param(ClientError({"Error": {"Code": "ValidationError", "Message": "Endpoint cuenta 123456789012 not found"}}, "InvokeEndpoint"), id="endpoint-inexistente"),
]


@pytest.mark.parametrize("error", CAIDAS)
def test_sagemaker_caido_responde_503_sin_filtrar_detalles(client, falso, error):
    falso.falla = error
    r = _post(client)
    assert r.status_code == 503, r.text
    for secreto in ("secreto", "/opt/ml", "123456789012", "amazonaws"):
        assert secreto not in r.text


# ------------------------------------------------------------------ SageMaker responde fuera de contrato -> 502
FUERA_DE_CONTRATO = [
    pytest.param(b"esto no es json", id="no-json"),
    pytest.param(b"{}", id="vacio"),
    pytest.param(b'{"probabilidad": 1.5, "aportes": {"a": 1.0}, "version_modelo": "x"}', id="probabilidad>1"),
    pytest.param(b'{"probabilidad": NaN, "aportes": {"a": 1.0}, "version_modelo": "x"}', id="probabilidad-NaN"),
    pytest.param(b'{"probabilidad": true, "aportes": {"a": 1.0}, "version_modelo": "x"}', id="probabilidad-booleana"),
    pytest.param(b'{"probabilidad": 0.1, "aportes": {}, "version_modelo": "x"}', id="sin-aportes"),
    pytest.param(b'{"probabilidad": 0.1, "aportes": {"a": "alto"}, "version_modelo": "x"}', id="aporte-texto"),
    pytest.param(b'{"probabilidad": 0.1, "aportes": [1, 2], "version_modelo": "x"}', id="aportes-lista"),
    pytest.param(b'{"probabilidad": 0.1, "aportes": {"a": 1.0}}', id="sin-version"),
    pytest.param(b'[1, 2, 3]', id="lista"),
]


@pytest.mark.parametrize("cuerpo", FUERA_DE_CONTRATO)
def test_respuesta_fuera_de_contrato_responde_502(client, falso, cuerpo):
    falso.cuerpo_crudo = cuerpo
    r = _post(client)
    assert r.status_code == 502, r.text
    assert "Traceback" not in r.text


def test_error_interno_no_filtra_detalles(client, falso, monkeypatch):
    def roto(*a, **k):
        raise RuntimeError("secreto interno /app/config/x")
    monkeypatch.setattr(main_module, "construir_features", roto)
    r = _post(client)
    assert r.status_code == 500
    assert "secreto" not in r.text and "/app" not in r.text
    assert falso.llamadas == []


def test_servicio_no_inicializado_responde_503(client, falso, monkeypatch):
    monkeypatch.setitem(main_module.estado, "listo", False)
    assert client.get("/health").status_code == 503
    assert _post(client).status_code == 503
    assert falso.llamadas == []


# ------------------------------------------------------------------ /health: reporta el estado del endpoint
def test_health_con_endpoint_in_service(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "healthy"
    assert d["endpoint"]["estado"] == "InService" and d["endpoint"]["nombre"] == main_module.ENDPOINT
    assert d["version_umbrales"]


def test_health_reporta_degradado_si_el_endpoint_no_esta_in_service(client, falso):
    """Por defecto /health NO se cae: si no, compose no levantaría el gateway y la política de reserva del dominio no serviría."""
    falso.estado = "Updating"
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degradado" and r.json()["endpoint"]["estado"] == "Updating"


def test_health_no_se_cae_si_describe_endpoint_falla(client, falso):
    falso.describe_falla = RuntimeError("secreto interno")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["endpoint"]["estado"] == "no_disponible"
    assert "secreto" not in r.text


def test_health_estricto_responde_503_si_no_esta_in_service(client, falso, monkeypatch):
    monkeypatch.setattr(main_module, "HEALTH_ESTRICTO", True)
    falso.estado = "Failed"
    assert client.get("/health").status_code == 503
    falso.estado = "InService"
    main_module.estado["salud"] = None            # vence la caché
    assert client.get("/health").status_code == 200


def test_health_cachea_describe_endpoint(client, falso):
    for _ in range(5):
        assert client.get("/health").status_code == 200
    assert falso.describe_llamadas == 1


def test_health_muestra_la_version_del_modelo_despues_de_predecir(client):
    assert client.get("/health").json()["version_modelo"] is None
    _post(client)
    assert client.get("/health").json()["version_modelo"] == VERSION_FALSA


# ------------------------------------------------------------------ boto3: timeouts acotados y SIN claves
def test_clientes_de_boto3_con_timeouts_y_sin_claves(client, falso):
    kw = falso.config_clientes["sagemaker-runtime"]
    assert kw["region_name"] == main_module.REGION
    cfg = kw["config"]
    assert cfg.connect_timeout == main_module.CONNECT_TIMEOUT_S and cfg.read_timeout == main_module.READ_TIMEOUT_S
    assert cfg.retries == {"max_attempts": 0}        # el reintento lo hace el gateway en la otra réplica
    assert "sagemaker" in falso.config_clientes
    for kwargs in falso.config_clientes.values():
        assert not any(("key" in k.lower()) or ("secret" in k.lower()) or ("token" in k.lower()) for k in kwargs), kwargs


def test_el_codigo_no_contiene_claves_de_aws():
    codigo = open(main_module.__file__, encoding="utf-8").read().lower()
    assert "aws_access_key" not in codigo and "aws_secret" not in codigo
