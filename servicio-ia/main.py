"""
servicio-ia/main.py · Microservicio de inferencia de fraude (Kipu)

Principios de esta versión:
  * UNA sola fuente de verdad para el tratamiento: se importan las funciones de datos/preparar.py
    (normalizar_transacciones, construir_variables), las mismas que generaron las variables de entrenamiento.
    Así no hay desfase entre lo que vio el modelo al entrenar y lo que recibe en línea.
  * El servicio NO consulta la base: el microservicio de dominio le envía la transacción y, opcionalmente,
    el perfil del usuario, el del dispositivo y las coincidencias de lista negra (contrato en TransaccionInput).
  * La decisión replica decidir_zona() de entrenar.py: de madrugada no hay zona "revisar" y la regla dura manda.
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

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RUTA_BASE = Path(__file__).resolve().parent
RUTA_MODELO = Path(os.getenv("RUTA_MODELO", RUTA_BASE / "model"))

# preparar.py vive en datos/ en el repositorio y junto a main.py dentro de la imagen
for _cand in (RUTA_BASE, RUTA_BASE.parent / "datos"):
    if (_cand / "preparar.py").exists():
        sys.path.insert(0, str(_cand))
        break
import preparar  # noqa: E402  (tratamiento compartido con el entrenamiento)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("servicio-ia")

estado: Dict[str, Any] = {}


def cargar_artefactos():
    try:
        opciones = ort.SessionOptions()
        # Un hilo por petición: el límite de CPU del contenedor es pequeño y así la latencia medida es honesta
        opciones.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
        opciones.inter_op_num_threads = 1

        prep = joblib.load(RUTA_MODELO / "preprocesador.joblib")
        sesion = ort.InferenceSession(str(RUTA_MODELO / "modelo.onnx"), sess_options=opciones, providers=["CPUExecutionProvider"])
        metadata = json.loads((RUTA_MODELO / "metadata.json").read_text(encoding="utf-8"))
        umbrales = json.loads((RUTA_MODELO / "umbrales.json").read_text(encoding="utf-8"))

        estado.update(prep=prep, sesion=sesion, metadata=metadata, umbrales=umbrales,
                      num=metadata["variables_numericas"],
                      features=metadata["variables_numericas"] + metadata["variables_categoricas"],
                      turno=metadata.get("zona_horaria", {}).get("turno", list(preparar.TURNO)),
                      listo=True, error=None)
        log.info("Modelo %s cargado (umbrales %s)", metadata["version"], umbrales["version"])
    except Exception as e:  # el servicio arranca igual; /health responde 503 y compose no lo marca healthy
        estado.update(listo=False, error=str(e))
        log.exception("No se pudieron cargar los artefactos del modelo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_artefactos()
    yield


app = FastAPI(title="Microservicio de Inferencia de Fraude - Kipu", version="2.1.0", lifespan=lifespan)


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


def _salidas_onnx(outs):
    """Misma selección por forma que entrenar.salidas_onnx: probabilidad (N,1) y aportes (N,k)."""
    p = next(o for o in outs if o.ndim == 2 and o.shape[1] == 1).ravel()
    c = next((o for o in outs if o.ndim == 2 and o.shape[1] > 1), None)
    return p, c


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health():
    if not estado.get("listo", False):
        raise HTTPException(status_code=503, detail="Modelo no listo")
    return {"status": "healthy", "servicio": "microservicio-ia", "version_modelo": estado["metadata"]["version"]}


@app.post("/predict", response_model=PrediccionResponse)
def predict(tx: TransaccionInput):
    if not estado.get("listo", False):
        raise HTTPException(status_code=503, detail="Servicio de modelo no inicializado")

    t0 = time.perf_counter()
    try:
        df_feat = construir_features(tx)
        X = df_feat[estado["features"]].copy()
        X[estado["num"]] = X[estado["num"]].astype(float)
        A = estado["prep"].transform(X).astype(np.float32)

        sesion = estado["sesion"]
        outs = sesion.run(None, {sesion.get_inputs()[0].name: A})
        p_serie, c_serie = _salidas_onnx(outs)
        p_cruda = float(p_serie[0])
        aportes = c_serie[0] if c_serie is not None else None

        meta, calib = estado["metadata"], estado["metadata"]["calibracion"]
        pc = float(np.clip(p_cruda, 1e-9, 1 - 1e-9))
        z = calib["a"] * math.log(pc / (1 - pc)) + calib["b"] + calib["log_razon_prior"]
        p_prod = float(1 / (1 + math.exp(-z)))

        dura = bool(df_feat["lista_negra_dura"].iloc[0])
        decision = decidir(p_prod, float(df_feat["hora_local"].iloc[0]), dura, estado["umbrales"]["umbrales"], estado["turno"])
        regla = None
        if dura:
            reglas = estado["umbrales"].get("reglas_duras") or ["lista_negra"]
            regla = reglas[0]

        razones: List[RazonExplicabilidad] = []
        if aportes is not None:
            delta = (aportes - np.array(meta["aportes_base"])) * calib["a"]
            for idx in np.argsort(-np.abs(delta))[:3]:
                nombre = meta["factores"][idx]
                razones.append(RazonExplicabilidad(
                    factor=nombre, descripcion=f"Aporte del factor {nombre}",
                    valor=_limpio(df_feat[nombre].iloc[0]) if nombre in df_feat.columns else None,
                    aporte_logodds=round(float(delta[idx]), 3),
                    efecto="sube el riesgo" if delta[idx] > 0 else "baja el riesgo"))

        return PrediccionResponse(
            transaccion_id=tx.transaccion_id, decision=decision, probabilidad_fraude=round(p_prod, 5),
            razones=razones, version_modelo=meta["version"], version_umbrales=estado["umbrales"]["version"],
            latencia_ms=round((time.perf_counter() - t0) * 1000, 2), regla_aplicada=regla)
    except EntradaInvalida as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    except Exception:
        log.exception("Fallo en la inferencia (transaccion_id=%s)", tx.transaccion_id)
        raise HTTPException(status_code=500, detail="Error interno durante la inferencia") from None
