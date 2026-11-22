"""servicio-dominio · API REST de negocio de Guardián Centinela (Kipu Pagos) · EP3.

Es el ÚNICO servicio que lee y escribe la base de datos. Para evaluar una transacción (RF1):
  1. valida la entrada (422 con el detalle; rechaza un PAN completo: RNF5 / PCI-DSS),
  2. lee de la base el perfil del usuario, el del dispositivo y las coincidencias de lista negra (RF8),
  3. llama al microservicio de IA (POST /predict), que invoca el endpoint de SageMaker,
  4. si la IA no responde a tiempo aplica la política de reserva (RNF3),
  5. registra la decisión con su entrada, versión del modelo y fecha (RF7, RNF7),
  6. si la decisión es revisar o bloquear, encola el pedido de resumen para el LLM (RF5). La respuesta NO espera al LLM:
     si la cola no está disponible, la predicción responde igual y queda en modo degradado.

Detrás del gateway, /api/... llega aquí sin el prefijo (ROOT_PATH=/api para que /docs muestre las rutas completas).
Todas las rutas de negocio exigen la cabecera X-API-Key (401 sin ella) y tienen límite de solicitudes por minuto
por cliente (429 al pasarlo). /health, /vivo y /docs son públicas.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

import httpx
import pika
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, field_validator

from anonimizar import mensaje_para_llm

DSN = os.getenv("DATABASE_URL", "postgresql://kipu:kipu@db:5432/kipu")
IA_URL = os.getenv("IA_URL", "http://ia:8000")                     # red interna: la ruta de IA no es pública
# Debe ser MAYOR que SAGEMAKER_READ_TIMEOUT_S de servicio-ia (5 s) para que un cold start alcance a responder,
# y MENOR que el proxy_read_timeout del gateway. Es un trade-off con el p99 < 100 ms (RNF1): se mide y se informa.
IA_TIMEOUT_S = float(os.getenv("IA_TIMEOUT_MS", "6000")) / 1000
POLITICA_RESERVA = os.getenv("POLITICA_RESERVA", "reglas")        # reglas | aprobar | bloquear (la define riesgo, ADR-3)
TIPO_CAMBIO_CLP_USD = float(os.getenv("TIPO_CAMBIO_CLP_USD", "950"))   # supuesto del equipo (el caso no lo entrega)
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/%2F")
COLA_LLM = os.getenv("COLA_LLM", "resumenes_llm")
COLA_FALLIDOS = f"{COLA_LLM}.fallidos"                          # los pedidos que quedaron en modo degradado
# Rate limiting (EP3 §2.5): solicitudes por minuto por cliente (clave + IP). Se sube en el .env para la prueba de Locust
LIMITE_POR_MINUTO = int(os.getenv("LIMITE_POR_MINUTO", "3000"))
CLAVES_API = [c.strip() for c in os.getenv("API_KEYS", "").split(",") if c.strip()]
PAN_RE = re.compile(r"\d{13,19}")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("servicio-dominio")

pool: ConnectionPool
http = httpx.Client(timeout=IA_TIMEOUT_S)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pool
    pool = ConnectionPool(DSN, min_size=1, max_size=int(os.getenv("DB_POOL_MAX", "8")), open=True,
                          kwargs={"row_factory": dict_row})
    if not CLAVES_API:
        log.warning("API_KEYS está vacía: todas las rutas protegidas responderán 401")
    yield
    pool.close()


app = FastAPI(title="Guardián Centinela · API de Kipu Pagos", version="3.0.0", lifespan=lifespan,
              root_path=os.getenv("ROOT_PATH", ""),
              description="Detección de fraude: decisión síncrona del modelo de SageMaker y resumen del LLM por la cola. "
                          "Las rutas de negocio exigen la cabecera X-API-Key.")


# ------------------------------------------------------------------ seguridad
def requiere_clave(x_api_key: Optional[str] = Header(default=None, description="Clave de API del sistema")):
    """401 si falta la clave o no coincide. compare_digest evita filtrar la clave por tiempos de respuesta."""
    if not x_api_key or not any(secrets.compare_digest(x_api_key, c) for c in CLAVES_API):
        raise HTTPException(401, "Falta la cabecera X-API-Key o la clave no es válida")


class LimitadorPorMinuto:
    """Ventana deslizante de 60 s en memoria, por cliente. Basta con un proceso de uvicorn por contenedor."""
    def __init__(self):
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def permitir(self, cliente: str, limite: int, ahora: float | None = None) -> bool:
        ahora = time.monotonic() if ahora is None else ahora
        with self._lock:
            q = self._hits.setdefault(cliente, deque())
            while q and ahora - q[0] >= 60:
                q.popleft()
            if len(q) >= limite:
                return False
            q.append(ahora)
            return True


limitador = LimitadorPorMinuto()


def limite_de_solicitudes(request: Request, x_api_key: Optional[str] = Header(default=None, include_in_schema=False)):
    """429 si el cliente pasa LIMITE_POR_MINUTO. La IP real llega en X-Forwarded-For desde Caddy y el gateway."""
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    if not limitador.permitir(f"{x_api_key}|{ip}", LIMITE_POR_MINUTO):
        log.warning("rate_limit cliente_ip=%s limite_por_minuto=%d", ip, LIMITE_POR_MINUTO)
        raise HTTPException(429, f"Demasiadas solicitudes: el límite es {LIMITE_POR_MINUTO} por minuto",
                            headers={"Retry-After": "60"})


# Primero la clave (401) y después el límite (429)
protegida = [Depends(requiere_clave), Depends(limite_de_solicitudes)]


# ------------------------------------------------------------------ contrato de entrada
class Transaccion(BaseModel):
    transaccion_id: Optional[str] = Field(default=None, max_length=40)
    usuario_id: str = Field(min_length=3, max_length=20)
    comercio_id: str = Field(min_length=2, max_length=20)
    dispositivo_id: Optional[str] = Field(default=None, max_length=40)
    monto: float = Field(gt=0, le=500_000_000)
    moneda: Literal["CLP", "USD"]
    canal: Literal["app", "web", "pos", "api"]
    mcc: str = Field(pattern=r"^\d{4}$")
    pais_transaccion: str = Field(min_length=2, max_length=2)
    ip: Optional[str] = None
    tarjeta_enmascarada: Optional[str] = None
    intentos_previos_1h: Optional[int] = Field(default=None, ge=0, le=100)
    timestamp: Optional[datetime] = None

    # extra="forbid": un campo que no está en el contrato (por ejemplo "tarjeta" con el número completo) -> 422 (RNF5)
    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [{
        "usuario_id": "KP301626", "comercio_id": "CM40294", "monto": 25000, "moneda": "CLP",
        "canal": "app", "mcc": "5411", "pais_transaccion": "CL"}]}}

    @field_validator("tarjeta_enmascarada")
    @classmethod
    def sin_pan(cls, v):  # RNF5 / RT4: el número de tarjeta nunca entra al sistema
        if v and PAN_RE.search(v.replace(" ", "").replace("-", "")):
            raise ValueError("Se recibió un número de tarjeta completo (PAN). Solo se acepta el enmascarado ****NNNN")
        return v

    @field_validator("pais_transaccion")
    @classmethod
    def mayusculas(cls, v):
        return v.upper()


# ------------------------------------------------------------------ lógica
def lista_negra(cur, t: Transaccion) -> tuple[bool, bool, list[dict]]:
    """RF8: coincidencias activas. Devuelve (dura, interna, razones). La dura (cumplimiento) obliga a bloquear (RF4)."""
    candidatos = [(tp, v) for tp, v in [("dispositivo", t.dispositivo_id), ("ip", t.ip),
                                         ("comercio", t.comercio_id), ("tarjeta", t.tarjeta_enmascarada)] if v]
    if not candidatos:
        return False, False, []
    cur.execute("SELECT tipo, valor, motivo, dura FROM lista_negra WHERE activa AND fecha_alta <= current_date AND (tipo, valor) IN ("
                + ",".join(["(%s,%s)"] * len(candidatos)) + ")", [x for par in candidatos for x in par])
    hits = cur.fetchall()
    razones = [{"tipo": "regla", "factor": f"lista_negra_{h['tipo']}",
                "descripcion": f"{h['tipo']} en lista negra ({h['motivo'] or 'sin motivo'}{', cumplimiento' if h['dura'] else ''})"}
               for h in hits]
    return any(h["dura"] for h in hits), any(not h["dura"] for h in hits), razones


def perfil_usuario(cur, usuario_id: str) -> Optional[dict]:
    cur.execute("""SELECT pais_usuario, segmento, verificado, telefono_verificado, nivel_kyc, dominio_desechable,
                          to_char(fecha_registro, 'YYYY-MM-DD') AS fecha_registro
                   FROM usuarios WHERE usuario_id = %s""", (usuario_id,))
    return cur.fetchone()


def perfil_dispositivo(cur, dispositivo_id: Optional[str]) -> Optional[dict]:
    if not dispositivo_id:
        return None
    cur.execute("SELECT tipo_dispositivo, pais_disp, emulador FROM dispositivos WHERE dispositivo_id = %s",
                (dispositivo_id,))
    return cur.fetchone()


def politica_reserva(t: Transaccion, monto_usd: float, dura: bool) -> tuple[str, list[dict]]:
    """RNF3 / ADR-3: qué se decide si la IA no responde. La regla dura siempre manda."""
    if dura:
        return "bloquear", [{"tipo": "reserva", "factor": "politica_reserva", "descripcion": "lista negra de cumplimiento"}]
    if POLITICA_RESERVA == "aprobar":
        return "aprobar", [{"tipo": "reserva", "factor": "politica_reserva", "descripcion": "aprobar todo"}]
    if POLITICA_RESERVA == "bloquear":
        return "bloquear", [{"tipo": "reserva", "factor": "politica_reserva", "descripcion": "bloquear todo"}]
    if monto_usd > 150 or (t.intentos_previos_1h or 0) >= 3 or t.pais_transaccion != "CL":
        return "revisar", [{"tipo": "reserva", "factor": "politica_reserva",
                            "descripcion": "motor de reglas: monto > USD 150, 3+ intentos o país extranjero"}]
    return "aprobar", [{"tipo": "reserva", "factor": "politica_reserva", "descripcion": "motor de reglas: sin alertas"}]


def publicar_en_cola(mensaje: dict) -> None:
    """Publica el pedido de resumen en RabbitMQ. Timeouts cortos: la cola nunca demora la predicción."""
    params = pika.URLParameters(RABBITMQ_URL)
    params.socket_timeout = 1
    params.blocked_connection_timeout = 1
    params.connection_attempts = 1
    con = pika.BlockingConnection(params)
    try:
        canal = con.channel()
        canal.queue_declare(queue=COLA_LLM, durable=True)
        canal.basic_publish(exchange="", routing_key=COLA_LLM, body=json.dumps(mensaje, default=str),
                            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"))
    finally:
        con.close()


def encolar_resumen(prediccion: dict) -> tuple[str, Optional[str]]:
    """Devuelve (estado_llm, motivo): pendiente si se encoló; degradado con el motivo si la cola falló."""
    try:
        publicar_en_cola(mensaje_para_llm(prediccion))
        return "pendiente", None
    except Exception as e:  # noqa: BLE001  (cualquier falla de la cola deja la predicción en modo degradado)
        log.warning("cola_no_disponible prediccion_id=%s error=%s", prediccion["prediccion_id"], e.__class__.__name__)
        return "degradado", f"cola: {e.__class__.__name__}"


# ------------------------------------------------------------------ salud
@app.get("/vivo", include_in_schema=False)
def vivo():
    """Healthcheck de Docker: solo la base. No depende del LLM ni de SageMaker (si dependiera, Docker botaría el dominio)."""
    try:
        with pool.connection(timeout=2) as con:
            con.execute("SELECT 1")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"base no disponible: {e.__class__.__name__}")
    return {"status": "ok"}


def _salud_base() -> dict:
    try:
        with pool.connection(timeout=2) as con:
            con.execute("SELECT 1")
        return {"estado": "ok"}
    except Exception as e:  # noqa: BLE001
        return {"estado": "caido", "detalle": e.__class__.__name__}


def _salud_sagemaker() -> dict:
    try:
        r = httpx.get(f"{IA_URL}/health", timeout=3)
        b = r.json()
        ep = b.get("endpoint", {})
        return {"estado": "ok" if r.status_code == 200 and ep.get("estado") == "InService" else "degradado",
                "endpoint": ep.get("nombre"), "estado_endpoint": ep.get("estado"), "version_modelo": b.get("version_modelo")}
    except Exception as e:  # noqa: BLE001
        return {"estado": "caido", "detalle": e.__class__.__name__}


def _salud_cola() -> dict:
    try:
        params = pika.URLParameters(RABBITMQ_URL)
        params.socket_timeout = 1
        params.connection_attempts = 1
        con = pika.BlockingConnection(params)
        try:
            canal = con.channel()
            pendientes = canal.queue_declare(queue=COLA_LLM, durable=True).method.message_count
            fallidos = canal.queue_declare(queue=COLA_FALLIDOS, durable=True).method.message_count
        finally:
            con.close()
        return {"estado": "ok", "pendientes": pendientes, "fallidos": fallidos}
    except Exception as e:  # noqa: BLE001
        return {"estado": "caido", "detalle": e.__class__.__name__}


def _salud_llm() -> dict:
    """El worker registra en estado_servicios cómo le fue con NVIDIA en su última llamada."""
    try:
        with pool.connection(timeout=2) as con:
            fila = con.execute("SELECT estado, detalle, actualizado FROM estado_servicios WHERE servicio = 'llm'").fetchone()
        if not fila:
            return {"estado": "sin_datos", "detalle": "el worker aún no ha llamado al LLM"}
        return {"estado": fila["estado"], "detalle": fila["detalle"], "actualizado": fila["actualizado"].isoformat()}
    except Exception as e:  # noqa: BLE001
        return {"estado": "desconocido", "detalle": e.__class__.__name__}


@app.get("/health", summary="Estado de cada pieza: base, SageMaker, cola y LLM (pública)")
def health():
    piezas = {"base": _salud_base(), "sagemaker": _salud_sagemaker(), "cola": _salud_cola(), "llm": _salud_llm()}
    criticas_ok = piezas["base"]["estado"] == "ok" and piezas["sagemaker"]["estado"] == "ok"
    todo_ok = criticas_ok and all(p["estado"] == "ok" for p in piezas.values())
    estado = "ok" if todo_ok else ("degradado" if piezas["base"]["estado"] == "ok" else "caido")
    return JSONResponse(status_code=200 if piezas["base"]["estado"] == "ok" else 503,
                        content={"status": estado, "servicio": "servicio-dominio", "piezas": piezas})


# ------------------------------------------------------------------ predicción
@app.post("/predict", dependencies=protegida, summary="Evalúa una transacción: aprobar, revisar o bloquear, con sus razones")
def predict(t: Transaccion):
    t0 = time.perf_counter()
    ts = (t.timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    tx_id = t.transaccion_id or f"TXN-{uuid.uuid4().hex[:12].upper()}"
    monto_usd = t.monto if t.moneda == "USD" else t.monto / TIPO_CAMBIO_CLP_USD

    with pool.connection() as con, con.cursor() as cur:
        usuario = perfil_usuario(cur, t.usuario_id)
        dispositivo = perfil_dispositivo(cur, t.dispositivo_id)
        dura, interna, razones_ln = lista_negra(cur, t)
        t_bd = time.perf_counter()

        # Contrato de servicio-ia/main.py (TransaccionInput). La tarjeta va solo enmascarada; el modelo no la usa.
        entrada_ia = {
            "transaccion_id": tx_id, "usuario_id": t.usuario_id, "comercio_id": t.comercio_id,
            "dispositivo_id": t.dispositivo_id, "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "monto": t.monto, "moneda": t.moneda, "canal": t.canal, "pais_transaccion": t.pais_transaccion,
            "mcc": int(t.mcc), "ip": t.ip, "intentos_previos_1h": t.intentos_previos_1h,
            "tarjeta_enmascarada": t.tarjeta_enmascarada,
            "usuario": usuario, "dispositivo": dispositivo,
            "lista_negra_interna": interna, "lista_negra_dura": dura,
        }

        prob, version_modelo, version_umbrales, falla = None, None, None, None
        try:
            r = http.post(f"{IA_URL}/predict", json=entrada_ia)
            if r.status_code == 422:           # entrada que pasó nuestra validación pero no la del modelo: es del cliente
                raise HTTPException(422, r.json().get("detail", "entrada inválida para el modelo"))
            r.raise_for_status()
            ia = r.json()
            decision, prob = ia["decision"], ia["probabilidad_fraude"]
            version_modelo, version_umbrales = ia["version_modelo"], ia["version_umbrales"]
            razones = razones_ln + [{"tipo": "modelo", **x} for x in ia["razones"]]
            origen = "regla_dura" if ia.get("regla_aplicada") or dura else "modelo"
        except httpx.TimeoutException:
            falla = "timeout: la IA no respondió a tiempo"
        except httpx.HTTPStatusError as e:
            falla = f"http_{e.response.status_code}: la IA no está disponible"
        except (httpx.HTTPError, KeyError, ValueError) as e:
            falla = f"error: {e.__class__.__name__}"
        t_ia = time.perf_counter()

        if falla:
            decision, razones = politica_reserva(t, monto_usd, dura)
            razones = razones_ln + razones + [{"tipo": "reserva", "factor": "ia_no_disponible", "descripcion": falla}]
            origen = "reserva"

        estado_llm = "pendiente" if decision in ("revisar", "bloquear") else "no_aplica"
        lat_ms = (time.perf_counter() - t0) * 1000
        cur.execute("""INSERT INTO predicciones (transaccion_id, usuario_id, dispositivo_id, entrada, probabilidad,
                           decision, origen, razones, version_modelo, version_umbrales, latencia_ms, monto_usd, estado_llm)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING prediccion_id, creado_utc""",
                    (tx_id, t.usuario_id, t.dispositivo_id, Jsonb(entrada_ia), prob, decision, origen,
                     Jsonb(razones), version_modelo, version_umbrales, round(lat_ms, 3), round(monto_usd, 4), estado_llm))
        reg = cur.fetchone()
        con.commit()

        respuesta = {"id": reg["prediccion_id"], "prediccion_id": reg["prediccion_id"], "transaccion_id": tx_id, "decision": decision,
                     "origen": origen, "probabilidad_fraude": prob, "razones": razones, "version_modelo": version_modelo,
                     "version_umbrales": version_umbrales, "usuario_conocido": usuario is not None,
                     "monto_usd": round(monto_usd, 2), "moneda": t.moneda, "canal": t.canal,
                     "pais_transaccion": t.pais_transaccion, "mcc": t.mcc, "estado_llm": estado_llm, "motivo_llm": None,
                     "latencia_ms": round(lat_ms, 2),
                     "desglose_ms": {"base_de_datos": round((t_bd - t0) * 1000, 2), "ia": round((t_ia - t_bd) * 1000, 2)},
                     "fecha_utc": reg["creado_utc"].isoformat()}

        # El LLM va por la cola, DESPUÉS de guardar: la decisión ya está tomada y no lo espera
        if estado_llm == "pendiente":
            estado_llm, motivo = encolar_resumen(respuesta)
            if estado_llm == "degradado":
                cur.execute("""UPDATE predicciones SET estado_llm = 'degradado', motivo_llm = %s, llm_actualizado = now()
                               WHERE prediccion_id = %s""", (motivo, reg["prediccion_id"]))
                con.commit()
            respuesta["estado_llm"], respuesta["motivo_llm"] = estado_llm, motivo

    # Una línea por predicción (CloudWatch). Sin tarjeta, nombre, RUT ni IP
    log.info("prediccion prediccion_id=%s transaccion_id=%s decision=%s origen=%s probabilidad=%s version_modelo=%s "
             "estado_llm=%s latencia_ms=%.1f ia_ms=%.1f", reg["prediccion_id"], tx_id, decision, origen, prob,
             version_modelo, respuesta["estado_llm"], lat_ms, (t_ia - t_bd) * 1000)
    return respuesta


@app.post("/evaluar", dependencies=protegida, include_in_schema=False)
def evaluar(t: Transaccion):
    """Ruta de EP2, se mantiene como alias de /predict."""
    return predict(t)


# ------------------------------------------------------------------ consultas
COLS_PRED = """prediccion_id AS id, prediccion_id, transaccion_id, usuario_id, decision, origen, probabilidad, razones,
               monto_usd, version_modelo, version_umbrales, latencia_ms, estado_llm, resumen_llm, motivo_llm,
               llm_actualizado, creado_utc"""


@app.get("/predicciones/{prediccion_id}", dependencies=protegida,
         summary="Una predicción guardada y, si ya llegó, el resumen del LLM")
def prediccion(prediccion_id: int):
    with pool.connection() as con:
        fila = con.execute(f"SELECT {COLS_PRED} FROM predicciones WHERE prediccion_id = %s", (prediccion_id,)).fetchone()
    if not fila:
        raise HTTPException(404, "No existe una predicción con ese id")
    return fila


@app.get("/predicciones", dependencies=protegida,
         summary="Últimas predicciones; con transaccion_id, todas las decisiones de esa transacción (RF7)")
def predicciones(limit: int = Query(20, ge=1, le=200), transaccion_id: Optional[str] = Query(None, max_length=40)):
    with pool.connection() as con:
        if transaccion_id:
            filas = con.execute(f"SELECT {COLS_PRED}, entrada FROM predicciones WHERE transaccion_id = %s ORDER BY prediccion_id",
                                (transaccion_id,)).fetchall()
            if not filas:
                raise HTTPException(404, "Sin decisiones registradas para esa transacción")
            return filas
        return con.execute(f"SELECT {COLS_PRED} FROM predicciones ORDER BY prediccion_id DESC LIMIT %s", (limit,)).fetchall()


@app.get("/revision", dependencies=protegida,
         summary="Cola de revisión priorizada por riesgo y monto (pérdida esperada = probabilidad × monto), con el resumen del LLM (RF5)")
def revision(limit: int = Query(50, ge=1, le=200)):
    """Primero lo que más dinero pone en riesgo. Las decisiones por reserva (sin probabilidad) van al final, por monto."""
    with pool.connection() as con:
        return con.execute(f"""SELECT {COLS_PRED}, round((probabilidad * monto_usd)::numeric, 2) AS perdida_esperada_usd
                               FROM predicciones
                               WHERE decision IN ('revisar','bloquear')
                               ORDER BY probabilidad * monto_usd DESC NULLS LAST, monto_usd DESC NULLS LAST, prediccion_id DESC
                               LIMIT %s""", (limit,)).fetchall()


@app.get("/dashboard", dependencies=protegida,
         summary="Transacciones evaluadas, distribución de decisiones, tasa de bloqueo por día (RF12) y estado del LLM")
def dashboard():
    with pool.connection() as con:
        total = con.execute("SELECT count(*) AS n FROM predicciones").fetchone()["n"]
        decisiones = con.execute("SELECT decision, count(*) AS n FROM predicciones GROUP BY 1 ORDER BY 1").fetchall()
        origenes = con.execute("SELECT origen, count(*) AS n FROM predicciones GROUP BY 1 ORDER BY 1").fetchall()
        por_dia = con.execute("""SELECT (creado_utc AT TIME ZONE 'America/Santiago')::date AS dia, count(*) AS evaluadas,
                                        count(*) FILTER (WHERE decision = 'bloquear') AS bloqueadas,
                                        round(100.0 * count(*) FILTER (WHERE decision = 'bloquear') / count(*), 2) AS tasa_bloqueo_pct
                                 FROM predicciones GROUP BY 1 ORDER BY 1 DESC LIMIT 30""").fetchall()
        llm = con.execute("""SELECT estado_llm, count(*) AS n FROM predicciones
                             WHERE decision IN ('revisar','bloquear') GROUP BY 1 ORDER BY 1""").fetchall()
        en_cola = con.execute("""SELECT count(*) AS n FROM predicciones WHERE decision IN ('revisar','bloquear')""").fetchone()["n"]
    return {"transacciones_evaluadas": total, "decisiones": decisiones, "origen": origenes,
            "bloqueo_por_dia": por_dia, "estado_llm": llm, "en_cola_de_revision": en_cola}


@app.get("/usuarios", dependencies=protegida, include_in_schema=False)
def usuarios(limit: int = Query(20, ge=1, le=200)):
    """Usuarios de ejemplo para el front, con un dispositivo y un comercio que hayan usado."""
    with pool.connection() as con:
        return con.execute("""
            SELECT u.usuario_id, u.segmento, u.nivel_kyc, t.dispositivo_id, t.comercio_id
            FROM usuarios u
            JOIN LATERAL (SELECT dispositivo_id, comercio_id FROM transacciones x
                          WHERE x.usuario_id = u.usuario_id AND x.comercio_id IS NOT NULL LIMIT 1) t ON true
            ORDER BY u.usuario_id LIMIT %s""", (limit,)).fetchall()


@app.get("/resumen", dependencies=protegida, include_in_schema=False)
def resumen():
    with pool.connection() as con:
        tablas = con.execute("""SELECT (SELECT count(*) FROM usuarios) AS usuarios,
                                       (SELECT count(*) FROM dispositivos) AS dispositivos,
                                       (SELECT count(*) FROM lista_negra WHERE activa) AS lista_negra_activa,
                                       (SELECT count(*) FROM transacciones) AS transacciones,
                                       (SELECT count(*) FROM predicciones) AS predicciones""").fetchone()
        decisiones = con.execute("SELECT decision, origen, count(*) AS n FROM predicciones GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    return {"tablas": tablas, "decisiones": decisiones}
