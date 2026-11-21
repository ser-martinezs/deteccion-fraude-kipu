"""
servicio-ia/main.py · Microservicio de IA de fraude (Kipu) · EP3: el modelo vive en un endpoint serverless de SageMaker

Esta imagen YA NO contiene el modelo. Su trabajo es:
  1. validar la transacción (422 si es inválida),
  2. construir las variables con datos/preparar.py (la MISMA fuente de verdad que usó el entrenamiento),
  3. invocar el endpoint de SageMaker con boto3 (sin claves: las credenciales vienen del perfil de la instancia EC2),
  4. aplicar decidir() con los umbrales de configuración (config/umbrales.json) y armar las razones con los 3 mayores aportes.

Contrato con el endpoint (aws/src/inference.py):
  entrada  {"variables": {<15 numéricas + 6 categóricas>}}      (null = faltante)
  salida   {"probabilidad": 0.0098, "aportes": {factor: log-odds}, "version_modelo": "3.0.0"}

Contrato con el dominio (sin cambios respecto de EP2): POST /predict -> PrediccionResponse.
Si SageMaker no responde, /predict contesta 503 y el dominio aplica su política de reserva (RNF3).

Configuración por variables de entorno (nada secreto):
  SAGEMAKER_ENDPOINT          nombre del endpoint                      (kipu-fraude-ep3)
  AWS_REGION                  región                                   (us-east-1)
  SAGEMAKER_CONNECT_TIMEOUT_S / SAGEMAKER_READ_TIMEOUT_S               (1 / 1.5)  ← deben ser menores que el read timeout del gateway (2 s)
                                                                         y que IA_TIMEOUT_MS del dominio, si se quiere ver la respuesta
  UMBRALES_PATH               ruta de umbrales.json                    (config/umbrales.json)
  HEALTH_TTL_S                caché de describe_endpoint en /health    (30)
  HEALTH_ESTRICTO             true = /health responde 503 si el endpoint no está InService (por defecto false: solo lo reporta)
"""
import json
import logging
import math
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

RUTA_BASE = Path(__file__).resolve().parent
RUTA_UMBRALES = Path(os.getenv("UMBRALES_PATH", RUTA_BASE / "config" / "umbrales.json"))
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "kipu-fraude-ep3")
CONNECT_TIMEOUT_S = float(os.getenv("SAGEMAKER_CONNECT_TIMEOUT_S", "1"))
READ_TIMEOUT_S = float(os.getenv("SAGEMAKER_READ_TIMEOUT_S", "5"))
HEALTH_TTL_S = float(os.getenv("HEALTH_TTL_S", "30"))
HEALTH_ESTRICTO = os.getenv("HEALTH_ESTRICTO", "false").strip().lower() in ("1", "true", "yes", "si", "sí")

# preparar.py vive en datos/ en el repositorio y junto a main.py dentro de la imagen
for _cand in (RUTA_BASE, RUTA_BASE.parent / "datos"):
    if (_cand / "preparar.py").exists():
        sys.path.insert(0, str(_cand))
        break
import preparar  # noqa: E402  (tratamiento compartido con el entrenamiento)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("servicio-ia")

estado: Dict[str, Any] = {}


class SageMakerNoDisponible(RuntimeError):
    """El endpoint no respondió (timeout, sin credenciales, throttling, endpoint caído): responde 503."""


class RespuestaInvalida(RuntimeError):
    """El endpoint respondió algo que no cumple el contrato: responde 502."""


def crear_clientes():
    """Clientes de boto3. Se crean sin claves: boto3 toma las credenciales del perfil de la instancia (LabInstanceProfile).
    Sin reintentos propios (max_attempts=0): el gateway ya reintenta en la otra réplica y así el tiempo total queda acotado."""
    reintentos = {"max_attempts": 0}
    rt = boto3.client("sagemaker-runtime", region_name=REGION,
                      config=Config(connect_timeout=CONNECT_TIMEOUT_S, read_timeout=READ_TIMEOUT_S, retries=reintentos))
    sm = boto3.client("sagemaker", region_name=REGION,
                      config=Config(connect_timeout=CONNECT_TIMEOUT_S, read_timeout=2, retries=reintentos))
    return rt, sm


def cargar_configuracion():
    """Umbrales (configuración, no parte del modelo) y clientes de AWS. Si los umbrales no sirven, /health responde 503."""
    try:
        umbrales = json.loads(RUTA_UMBRALES.read_text(encoding="utf-8"))
        u = {k: float(umbrales["umbrales"][k]) for k in ("revisar_dia", "bloquear_dia", "bloquear_noche")}
        if not (0.0 <= u["revisar_dia"] <= u["bloquear_dia"] <= 1.0 and 0.0 <= u["bloquear_noche"] <= 1.0):
            raise ValueError(f"umbrales incoherentes: {u}")
        umbrales["umbrales"] = u
        rt, sm = crear_clientes()
        estado.update(umbrales=umbrales, turno=list(preparar.TURNO), num=list(preparar.FEATURES_NUM),
                      cat=list(preparar.FEATURES_CAT), runtime=rt, sm=sm, version_modelo=None, salud=None, listo=True, error=None)
        log.info("Configuración cargada: umbrales %s · endpoint %s (%s)", umbrales.get("version"), ENDPOINT, REGION)
    except Exception as e:  # el servicio arranca igual; /health responde 503 y compose no lo marca healthy
        estado.update(listo=False, error=str(e))
        log.exception("No se pudo cargar la configuración del servicio")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_configuracion()
    yield


app = FastAPI(title="Microservicio de IA de Fraude - Kipu", version="3.0.0", lifespan=lifespan)


# -------------------------------------------------------------
# Contrato (esquemas Pydantic)
# -------------------------------------------------------------
class PerfilUsuario(BaseModel):
    """Misma forma que la tabla `usuarios` tratada (datos/tratado/usuarios.csv)."""
    pais_usuario: Optional[str] = None
    segmento: Optional[Literal["persona", "comercio_pequeno", "comercio_grande"]] = None
    verificado: Optional[float] = Field(None, ge=0, le=1)
    telefono_verificado: Optional[float] = Field(None, ge=0, le=1)
    nivel_kyc: Optional[float] = Field(None, ge=0, le=3)
    dominio_desechable: Optional[float] = Field(None, ge=0, le=1)
    fecha_registro: Optional[str] = Field(None, description="YYYY-MM-DD")


class PerfilDispositivo(BaseModel):
    """Misma forma que la tabla `dispositivos` tratada."""
    tipo_dispositivo: Optional[Literal["android", "ios", "web", "pos"]] = None
    pais_disp: Optional[str] = None
    emulador: Optional[float] = Field(None, ge=0, le=1)


class TransaccionInput(BaseModel):
    transaccion_id: str
    usuario_id: str
    comercio_id: str
    dispositivo_id: Optional[str] = None
    timestamp: str = Field(..., description="ISO8601 con zona (Z, -04:00), 'YYYY-MM-DD HH:MM:SS' o 'dd/mm/YYYY HH:MM'. Sin zona = hora de Chile")
    monto: Any = Field(..., description="Número JSON (decimal con punto) o texto ('$1.200' = 1200 CLP, '45,50')")
    moneda: Optional[str] = Field(None, description="CLP o USD; si falta se infiere como en el entrenamiento")
    canal: str
    pais_transaccion: str
    mcc: Any
    latitud: Optional[float] = None
    longitud: Optional[float] = None
    ip: Optional[str] = None
    intentos_previos_1h: Optional[float] = Field(None, ge=0, description="Si no se conoce, no enviar: el modelo lo trata como faltante")
    tarjeta_enmascarada: Optional[str] = None
    # Enriquecimiento que entrega el microservicio de dominio (lo lee de la base)
    usuario: Optional[PerfilUsuario] = None
    dispositivo: Optional[PerfilDispositivo] = None
    lista_negra_interna: bool = Field(False, description="Coincide con una entrada activa NO dura de la lista negra")
    lista_negra_dura: bool = Field(False, description="Coincide con una entrada dura (solicitud de cumplimiento): bloqueo obligatorio")


class RazonExplicabilidad(BaseModel):
    factor: str
    descripcion: str
    valor: Any
    aporte_logodds: float
    efecto: str


class PrediccionResponse(BaseModel):
    transaccion_id: str
    decision: str = Field(..., description="aprobar, revisar o bloquear")
    probabilidad_fraude: float
    razones: List[RazonExplicabilidad]
    version_modelo: str
    version_umbrales: str
    latencia_ms: float
    regla_aplicada: Optional[str] = Field(None, description="Regla dura que forzó la decisión, si la hubo")


# -------------------------------------------------------------
# Preprocesamiento: reutiliza datos/preparar.py
# -------------------------------------------------------------
class EntradaInvalida(ValueError):
    """La transacción no se puede interpretar: responde 422 (no 500)."""


def _monto_a_texto(m: Any) -> str:
    if m is None or isinstance(m, bool):
        raise EntradaInvalida("monto: se esperaba un número o un texto")
    if isinstance(m, (int, float)):
        if not math.isfinite(m):
            raise EntradaInvalida("monto: no es un número finito")
        # En un número JSON el punto es SIEMPRE decimal. Con coma, preparar.parse_monto no lo confunde con separador de miles
        return str(m).replace(".", ",")
    return str(m)


def _mcc_a_texto(m: Any) -> str:
    try:
        return str(int(float(str(m).strip())))
    except (TypeError, ValueError):
        raise EntradaInvalida("mcc: debe ser un código numérico") from None


def _normalizar(tx: TransaccionInput) -> pd.DataFrame:
    cruda = pd.DataFrame([{
        "transaccion_id": tx.transaccion_id, "usuario_id": tx.usuario_id, "comercio_id": tx.comercio_id,
        "dispositivo_id": tx.dispositivo_id, "timestamp": tx.timestamp, "monto": _monto_a_texto(tx.monto),
        "moneda": tx.moneda, "canal": tx.canal, "pais_transaccion": tx.pais_transaccion, "mcc": _mcc_a_texto(tx.mcc),
        "latitud": tx.latitud, "longitud": tx.longitud, "ip": tx.ip, "tarjeta_enmascarada": tx.tarjeta_enmascarada,
        "intentos_previos_1h": tx.intentos_previos_1h,
    }])
    try:
        n = preparar.normalizar_transacciones(cruda)      # R03-R11: las mismas reglas del entrenamiento
    except preparar.ErrorDeDatos as e:
        raise EntradaInvalida(str(e).split(";")[0]) from None   # categoría desconocida (canal, país)
    if n["ts_utc"].isna().iloc[0]:
        raise EntradaInvalida("timestamp: formato no reconocido (ISO8601 con zona, 'YYYY-MM-DD HH:MM:SS' o 'dd/mm/YYYY HH:MM')")
    if n["monto_valor"].isna().iloc[0]:
        raise EntradaInvalida("monto: no se pudo interpretar")
    if n["moneda_final"].iloc[0] not in ("CLP", "USD"):
        raise EntradaInvalida("moneda: solo se acepta CLP o USD")
    return n


def _df_usuario(usuario_id: str, p: Optional[PerfilUsuario]) -> pd.DataFrame:
    p = p or PerfilUsuario()
    fecha = pd.NaT
    if p.fecha_registro:
        fecha = pd.to_datetime(p.fecha_registro, errors="coerce")
        if pd.isna(fecha):
            raise EntradaInvalida("usuario.fecha_registro: formato inválido (YYYY-MM-DD)")
        if fecha.tzinfo is not None:
            fecha = fecha.tz_convert("UTC").tz_localize(None)
    fila = {"usuario_id": usuario_id, "pais_usuario": p.pais_usuario.strip().upper() if p.pais_usuario else None,
            "segmento": p.segmento, "verificado": p.verificado, "telefono_verificado": p.telefono_verificado,
            "nivel_kyc": p.nivel_kyc, "dominio_desechable": p.dominio_desechable, "fecha_registro": fecha}
    tipos = {"verificado": float, "telefono_verificado": float, "nivel_kyc": float, "dominio_desechable": float, "fecha_registro": "datetime64[ns]"}
    return pd.DataFrame([fila]).astype(tipos)


def _df_dispositivo(dispositivo_id: Optional[str], p: Optional[PerfilDispositivo]) -> pd.DataFrame:
    p = p or PerfilDispositivo()
    if dispositivo_id is None:       # sin dispositivo no hay perfil: las variables quedan nulas (como en el entrenamiento)
        p = PerfilDispositivo()
    fila = {"dispositivo_id": dispositivo_id, "tipo_dispositivo": p.tipo_dispositivo,
            "pais_disp": p.pais_disp.strip().upper() if p.pais_disp else None, "emulador": p.emulador}
    return pd.DataFrame([fila]).astype({"emulador": float})


# La lista negra ya la cruzó el dominio contra la base: aquí solo se necesita una tabla vacía con los tipos correctos
_LISTA_NEGRA_VACIA = pd.DataFrame({"tipo": pd.Series(dtype="object"), "valor": pd.Series(dtype="object"),
                                   "fecha_alta": pd.Series(dtype="datetime64[ns]"), "activa": pd.Series(dtype="bool"),
                                   "dura": pd.Series(dtype="bool")})


def construir_features(tx: TransaccionInput) -> pd.DataFrame:
    """Una fila con todas las variables del modelo, calculadas con el mismo código que el entrenamiento."""
    n = _normalizar(tx)
    x = preparar.construir_variables(n, _df_usuario(tx.usuario_id, tx.usuario),
                                     _df_dispositivo(tx.dispositivo_id, tx.dispositivo), _LISTA_NEGRA_VACIA)
    x["lista_negra_interna"] = 1.0 if tx.lista_negra_interna else 0.0
    x["lista_negra_dura"] = bool(tx.lista_negra_dura)
    return x


def decidir(p: float, hora: float, dura: bool, umb: Dict[str, float], turno) -> str:
    """Misma lógica que decidir_zona() de entrenar.py: la regla dura manda; fuera de turno no hay 'revisar'."""
    if dura:
        return "bloquear"
    en_turno = turno[0] <= hora < turno[1]
    umbral_bloqueo = umb["bloquear_dia"] if en_turno else umb["bloquear_noche"]
    umbral_revision = umb["revisar_dia"] if en_turno else math.inf
    if p >= umbral_bloqueo:
        return "bloquear"
    if p >= umbral_revision:
        return "revisar"
    return "aprobar"


def _limpio(v: Any) -> Any:
    """Valor apto para JSON: NaN/NaT -> None y tipos de numpy/pandas -> tipos de Python."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(v) else str(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


# -------------------------------------------------------------
# Invocación de SageMaker
# -------------------------------------------------------------
def construir_entrada(df_feat: pd.DataFrame) -> Dict[str, Any]:
    """Variables del modelo en JSON: numéricas como número, categóricas como texto, faltante = null."""
    fila = df_feat.iloc[0]
    variables: Dict[str, Any] = {}
    for c in estado["num"]:
        variables[c] = None if pd.isna(fila[c]) else float(fila[c])
    for c in estado["cat"]:
        variables[c] = None if pd.isna(fila[c]) else str(fila[c])
    return {"variables": variables}


def validar_respuesta(cuerpo: bytes) -> Dict[str, Any]:
    """Comprueba el contrato {probabilidad, aportes, version_modelo}; cualquier otra cosa es RespuestaInvalida."""
    def numero(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    try:
        d = json.loads(cuerpo)
        p, aportes, version = d["probabilidad"], d["aportes"], d["version_modelo"]
    except (ValueError, KeyError, TypeError):
        raise RespuestaInvalida("no es el JSON del contrato") from None
    if not (numero(p) and 0.0 <= p <= 1.0):
        raise RespuestaInvalida("probabilidad fuera de [0, 1]")
    if not (isinstance(aportes, dict) and aportes and all(numero(v) for v in aportes.values())):
        raise RespuestaInvalida("aportes inválidos")
    if not (isinstance(version, str) and version):
        raise RespuestaInvalida("version_modelo ausente")
    return {"probabilidad": float(p), "aportes": {str(k): float(v) for k, v in aportes.items()}, "version_modelo": version}


def invocar_sagemaker(entrada: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = estado["runtime"].invoke_endpoint(EndpointName=ENDPOINT, ContentType="application/json", Accept="application/json",
                                              Body=json.dumps(entrada, allow_nan=False))
        cuerpo = r["Body"].read()
    except NoCredentialsError:
        log.error("SageMaker: no encuentro credenciales de AWS (¿perfil LabInstanceProfile en la EC2 y hop limit 2 para Docker?)")
        raise SageMakerNoDisponible() from None
    except (BotoCoreError, ClientError) as e:      # timeouts, conexión, throttling, ModelError, endpoint inexistente...
        codigo = e.response["Error"]["Code"] if isinstance(e, ClientError) else type(e).__name__
        log.error("SageMaker no respondió: %s", codigo)
        raise SageMakerNoDisponible() from None
    return validar_respuesta(cuerpo)


def estado_endpoint() -> Dict[str, Any]:
    """describe_endpoint con caché de HEALTH_TTL_S segundos (el healthcheck de Docker consulta cada 10 s)."""
    ahora, cache = time.monotonic(), estado.get("salud")
    if cache and ahora - cache[0] < HEALTH_TTL_S:
        return cache[1]
    try:
        res = {"estado": estado["sm"].describe_endpoint(EndpointName=ENDPOINT)["EndpointStatus"]}
    except Exception as e:                          # sin permisos, sin red, endpoint inexistente...
        codigo = e.response["Error"]["Code"] if isinstance(e, ClientError) else type(e).__name__
        res = {"estado": "no_disponible", "error": codigo}
    estado["salud"] = (ahora, res)
    return res


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health():
    if not estado.get("listo", False):
        raise HTTPException(status_code=503, detail="Servicio no inicializado")
    ep = estado_endpoint()
    en_servicio = ep["estado"] == "InService"
    cuerpo = {"status": "healthy" if en_servicio else "degradado", "servicio": "microservicio-ia",
              "endpoint": {"nombre": ENDPOINT, "region": REGION, **ep}, "version_modelo": estado.get("version_modelo"),
              "version_umbrales": estado["umbrales"].get("version")}
    if HEALTH_ESTRICTO and not en_servicio:
        return JSONResponse(status_code=503, content=cuerpo)
    return cuerpo


@app.post("/predict", response_model=PrediccionResponse)
def predict(tx: TransaccionInput):
    if not estado.get("listo", False):
        raise HTTPException(status_code=503, detail="Servicio de IA no inicializado")

    t0 = time.perf_counter()
    try:
        df_feat = construir_features(tx)
        sm = invocar_sagemaker(construir_entrada(df_feat))
        estado["version_modelo"] = sm["version_modelo"]

        dura = bool(df_feat["lista_negra_dura"].iloc[0])
        decision = decidir(sm["probabilidad"], float(df_feat["hora_local"].iloc[0]), dura, estado["umbrales"]["umbrales"], estado["turno"])
        regla = None
        if dura:
            reglas = estado["umbrales"].get("reglas_duras") or ["lista_negra"]
            regla = reglas[0]

        razones: List[RazonExplicabilidad] = []
        for nombre, aporte in sorted(sm["aportes"].items(), key=lambda kv: -abs(kv[1]))[:3]:
            razones.append(RazonExplicabilidad(
                factor=nombre, descripcion=f"Aporte del factor {nombre}",
                valor=_limpio(df_feat[nombre].iloc[0]) if nombre in df_feat.columns else None,
                aporte_logodds=round(aporte, 3), efecto="sube el riesgo" if aporte > 0 else "baja el riesgo"))

        return PrediccionResponse(
            transaccion_id=tx.transaccion_id, decision=decision, probabilidad_fraude=round(sm["probabilidad"], 5),
            razones=razones, version_modelo=sm["version_modelo"], version_umbrales=estado["umbrales"]["version"],
            latencia_ms=round((time.perf_counter() - t0) * 1000, 2), regla_aplicada=regla)
    except EntradaInvalida as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    except SageMakerNoDisponible:
        raise HTTPException(status_code=503, detail="El modelo (endpoint de SageMaker) no está disponible") from None
    except RespuestaInvalida as e:
        log.error("Respuesta de SageMaker fuera de contrato (transaccion_id=%s): %s", tx.transaccion_id, e)
        raise HTTPException(status_code=502, detail="El modelo devolvió una respuesta inesperada") from None
    except HTTPException:
        raise
    except Exception:
        log.exception("Fallo en la inferencia (transaccion_id=%s)", tx.transaccion_id)
        raise HTTPException(status_code=500, detail="Error interno durante la inferencia") from None
