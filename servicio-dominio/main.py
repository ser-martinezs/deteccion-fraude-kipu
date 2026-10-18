"""servicio-dominio · API REST de negocio de Guardián Centinela (Kipu Pagos).

Es el ÚNICO servicio que lee y escribe la base de datos. Para evaluar una transacción (RF1):
  1. valida la entrada (rechaza un PAN completo: RNF5 / PCI-DSS),
  2. lee de la base el perfil del usuario, el del dispositivo y las coincidencias de lista negra (RF8),
  3. llama al microservicio de IA (POST /predict) con el contrato de servicio-ia/main.py (TransaccionInput),
  4. si la IA no responde a tiempo aplica la política de reserva (RNF3),
  5. registra la decisión con su entrada, versión del modelo y fecha (RF7, RNF7).
"""
from __future__ import annotations

import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, field_validator

DSN = os.getenv("DATABASE_URL", "postgresql://kipu:kipu@db:5432/kipu")
IA_URL = os.getenv("IA_URL", "http://gateway:8080/api/ia")       # la IA se consume a través del gateway (balanceo)
IA_TIMEOUT_S = float(os.getenv("IA_TIMEOUT_MS", "500")) / 1000
POLITICA_RESERVA = os.getenv("POLITICA_RESERVA", "reglas")        # reglas | aprobar | bloquear (la define riesgo, ADR-3)
PAN_RE = re.compile(r"\d{13,19}")

pool: ConnectionPool
http = httpx.Client(timeout=IA_TIMEOUT_S)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pool
    pool = ConnectionPool(DSN, min_size=1, max_size=int(os.getenv("DB_POOL_MAX", "8")), open=True,
                          kwargs={"row_factory": dict_row})
    yield
    pool.close()


app = FastAPI(title="Guardián Centinela · servicio-dominio", version="1.0.0", lifespan=lifespan,
              root_path=os.getenv("ROOT_PATH", ""))


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
                "descripcion": f"{h['valor']} vetado ({h['motivo'] or 'sin motivo'}{', cumplimiento' if h['dura'] else ''})"}
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


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    try:
        with pool.connection(timeout=2) as con:
            con.execute("SELECT 1")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"base no disponible: {e.__class__.__name__}")
    return {"status": "healthy", "servicio": "servicio-dominio"}


@app.post("/evaluar")
def evaluar(t: Transaccion):
    t0 = time.perf_counter()
    ts = (t.timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    tx_id = t.transaccion_id or f"TXN-{uuid.uuid4().hex[:12].upper()}"
    monto_usd = t.monto if t.moneda == "USD" else t.monto / 950

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

        lat_ms = (time.perf_counter() - t0) * 1000
        cur.execute("""INSERT INTO predicciones (transaccion_id, usuario_id, dispositivo_id, entrada, probabilidad,
                           decision, origen, razones, version_modelo, version_umbrales, latencia_ms)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING prediccion_id, creado_utc""",
                    (tx_id, t.usuario_id, t.dispositivo_id, Jsonb(entrada_ia), prob, decision, origen,
                     Jsonb(razones), version_modelo, version_umbrales, round(lat_ms, 3)))
        reg = cur.fetchone()
        con.commit()

    return {"prediccion_id": reg["prediccion_id"], "transaccion_id": tx_id, "decision": decision, "origen": origen,
            "probabilidad_fraude": prob, "razones": razones, "version_modelo": version_modelo,
            "version_umbrales": version_umbrales, "usuario_conocido": usuario is not None,
            "latencia_ms": round(lat_ms, 2),
            "desglose_ms": {"base_de_datos": round((t_bd - t0) * 1000, 2), "ia": round((t_ia - t_bd) * 1000, 2)},
            "fecha_utc": reg["creado_utc"].isoformat()}


@app.get("/predicciones")
def predicciones(limit: int = Query(20, ge=1, le=200)):
    with pool.connection() as con:
        return con.execute("""SELECT prediccion_id, transaccion_id, usuario_id, decision, origen, probabilidad,
                                     version_modelo, version_umbrales, latencia_ms, creado_utc
                              FROM predicciones ORDER BY prediccion_id DESC LIMIT %s""", (limit,)).fetchall()


@app.get("/predicciones/{transaccion_id}")
def reproducir(transaccion_id: str):
    """RF7: qué modelo, qué datos y qué umbrales decidieron una transacción."""
    with pool.connection() as con:
        filas = con.execute("SELECT * FROM predicciones WHERE transaccion_id = %s ORDER BY prediccion_id",
                            (transaccion_id,)).fetchall()
    if not filas:
        raise HTTPException(404, "Sin decisiones registradas para esa transacción")
    return filas


@app.get("/usuarios")
def usuarios(limit: int = Query(20, ge=1, le=200)):
    """Usuarios de ejemplo para el front, con un dispositivo y un comercio que hayan usado."""
    with pool.connection() as con:
        return con.execute("""
            SELECT u.usuario_id, u.segmento, u.nivel_kyc, t.dispositivo_id, t.comercio_id
            FROM usuarios u
            JOIN LATERAL (SELECT dispositivo_id, comercio_id FROM transacciones x
                          WHERE x.usuario_id = u.usuario_id AND x.comercio_id IS NOT NULL LIMIT 1) t ON true
            ORDER BY u.usuario_id LIMIT %s""", (limit,)).fetchall()


@app.get("/resumen")
def resumen():
    with pool.connection() as con:
        tablas = con.execute("""SELECT (SELECT count(*) FROM usuarios) AS usuarios,
                                       (SELECT count(*) FROM dispositivos) AS dispositivos,
                                       (SELECT count(*) FROM lista_negra WHERE activa) AS lista_negra_activa,
                                       (SELECT count(*) FROM transacciones) AS transacciones,
                                       (SELECT count(*) FROM predicciones) AS predicciones""").fetchone()
        decisiones = con.execute("SELECT decision, origen, count(*) AS n FROM predicciones GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    return {"tablas": tablas, "decisiones": decisiones}
