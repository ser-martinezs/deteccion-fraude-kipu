import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RUTA_BASE = Path(__file__).resolve().parent
RUTA_MODELO = RUTA_BASE / "model"

app = FastAPI(
    title="Microservicio de Inferencia de Fraude - Kipu",
    version="2.0.0"
)

# -------------------------------------------------------------
# Esquemas Pydantic
# -------------------------------------------------------------
class TransaccionInput(BaseModel):
    transaccion_id: str
    usuario_id: str
    comercio_id: str
    dispositivo_id: Optional[str] = None
    timestamp: str = Field(..., description="ISO8601 o fecha legible")
    monto: Any = Field(..., description="Monto en formato numérico o string ($1.200, 45.50)")
    moneda: Optional[str] = None
    canal: str
    pais_transaccion: str
    mcc: Any
    latitud: Optional[float] = None
    longitud: Optional[float] = None
    ip: Optional[str] = None
    intentos_previos_1h: float = 0.0
    tarjeta_enmascarada: Optional[str] = None

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

# -------------------------------------------------------------
# Carga de artefactos en memoria
# -------------------------------------------------------------
estado: Dict[str, Any] = {}

def cargar_artefactos():
    try:
        prep = joblib.load(RUTA_MODELO / "preprocesador.joblib")
        sesion = ort.InferenceSession(str(RUTA_MODELO / "modelo.onnx"), providers=["CPUExecutionProvider"])
        metadata = json.loads((RUTA_MODELO / "metadata.json").read_text(encoding="utf-8"))
        umbrales = json.loads((RUTA_MODELO / "umbrales.json").read_text(encoding="utf-8"))

        estado["prep"] = prep
        estado["sesion"] = sesion
        estado["metadata"] = metadata
        estado["umbrales"] = umbrales
        estado["listo"] = True
    except Exception as e:
        estado["listo"] = False
        estado["error"] = str(e)

@app.on_event("startup")
def startup_event():
    cargar_artefactos()

# -------------------------------------------------------------
# Lógica de cálculo de variables (alineada con preparar.py)
# -------------------------------------------------------------
def parse_monto(m):
    s = str(m).strip().replace("$", "")
    miles = bool(re.match(r"^-?\d{1,3}(?:\.\d{3})+$", s))
    if miles:
        s = s.replace(".", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s), miles
    except ValueError:
        return 0.0, False

def transformar_entrada(tx_dict: dict) -> pd.DataFrame:
    # 1. Parse de timestamp a hora local de Chile
    try:
        ts = pd.to_datetime(tx_dict["timestamp"], utc=True)
        # Desfase fijo UTC-4 (hora Chile según preparar.py)
        ts_chile = ts - pd.Timedelta(hours=4)
        hora_local = float(ts_chile.hour)
    except Exception:
        hora_local = 12.0
    es_madrugada = 1.0 if hora_local < 6 else 0.0

    # 2. Parse de monto y moneda
    valor, miles = parse_monto(tx_dict.get("monto", 0))
    moneda = str(tx_dict.get("moneda") or "").strip().upper()
    if not moneda:
        moneda = "CLP" if (miles or abs(valor) >= 500) else "USD"
    monto_usd = abs(valor) / 950.0 if moneda == "CLP" else abs(valor)
    log_monto_usd = float(np.log1p(monto_usd))

    # 3. IP ausente/inválida
    ip = tx_dict.get("ip")
    ip_inv = 1.0
    if ip and not pd.isna(ip):
        p = str(ip).split(".")
        if len(p) == 4 and all(o.isdigit() and 0 <= int(o) <= 255 for o in p):
            ip_inv = 0.0

    # 4. Construcción del DataFrame con todas las columnas del preprocesador
    fila = {
        # Variables numéricas
        "log_monto_usd": log_monto_usd,
        "hora_local": hora_local,
        "es_madrugada": es_madrugada,
        "intentos_previos_1h": float(tx_dict.get("intentos_previos_1h") or 0.0),
        "nivel_kyc": np.nan,  # Se imputa automáticamente con la mediana
        "verificado": np.nan,
        "telefono_verificado": np.nan,
        "dominio_desechable": np.nan,
        "edad_cuenta_dias": np.nan,
        "emulador": np.nan,
        "pais_distinto_usuario": np.nan,
        "pais_distinto_dispositivo": np.nan,
        "ip_ausente_o_invalida": ip_inv,
        "dispositivo_ausente": 1.0 if not tx_dict.get("dispositivo_id") else 0.0,
        "lista_negra_interna": 0.0,
        # Variables categóricas
        "moneda_final": moneda,
        "canal": str(tx_dict.get("canal", "app")).lower(),
        "pais_tx": str(tx_dict.get("pais_transaccion", "CL")).upper(),
        "segmento": "persona",
        "tipo_dispositivo": "android",
        "mcc": str(tx_dict.get("mcc", "5411"))
    }
    return pd.DataFrame([fila])

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health():
    if not estado.get("listo", False):
        raise HTTPException(status_code=503, detail=f"Modelo no listo: {estado.get('error', 'Cargando')}")
    return {
        "status": "healthy",
        "servicio": "microservicio-ia",
        "version_modelo": estado["metadata"]["version"]
    }

@app.post("/predict", response_model=PrediccionResponse)
def predict(tx: TransaccionInput):
    if not estado.get("listo", False):
        raise HTTPException(status_code=503, detail="Servicio de modelo no inicializado")

    t0 = time.perf_counter()

    try:
        # Pydantic V2 usa model_dump(), fallback a dict()
        tx_dict = tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()
        df_feat = transformar_entrada(tx_dict)

        # Preprocesar
        A = estado["prep"].transform(df_feat).astype(np.float32)

        # Inferencia ONNX
        sesion = estado["sesion"]
        nombre_in = sesion.get_inputs()[0].name
        outs = sesion.run(None, {nombre_in: A})

        p_cruda = float(outs[0].ravel()[0])
        aportes = outs[1][0] if len(outs) > 1 else None

        # Calibración Platt a producción
        calib = estado["metadata"]["calibracion"]
        logit_p = np.log(np.clip(p_cruda, 1e-9, 1 - 1e-9) / (1 - np.clip(p_cruda, 1e-9, 1 - 1e-9)))
        z = calib["a"] * logit_p + calib["b"] + calib["log_razon_prior"]
        p_prod = float(1 / (1 + np.exp(-z)))

        # Decisión según umbrales de turno día
        umb = estado["umbrales"]["umbrales"]
        if p_prod >= umb["bloquear_dia"]:
            decision = "bloquear"
        elif p_prod >= umb["revisar_dia"]:
            decision = "revisar"
        else:
            decision = "aprobar"

        # Explicabilidad de los aportes
        razones = []
        if aportes is not None:
            factores = estado["metadata"]["factores"]
            base_aportes = np.array(estado["metadata"]["aportes_base"])
            delta = (aportes - base_aportes) * calib["a"]
            top_idx = np.argsort(-np.abs(delta))[:3]
            for idx in top_idx:
                f_nom = factores[idx]
                razones.append(RazonExplicabilidad(
                    factor=f_nom,
                    descripcion=f"Aporte del factor {f_nom}",
                    valor=df_feat[f_nom].iloc[0] if f_nom in df_feat.columns else None,
                    aporte_logodds=round(float(delta[idx]), 3),
                    efecto="sube el riesgo" if delta[idx] > 0 else "baja el riesgo"
                ))

        latencia = round((time.perf_counter() - t0) * 1000, 2)

        return PrediccionResponse(
            transaccion_id=tx.transaccion_id,
            decision=decision,
            probabilidad_fraude=round(p_prod, 5),
            razones=razones,
            version_modelo=estado["metadata"]["version"],
            version_umbrales=estado["umbrales"]["version"],
            latencia_ms=latencia
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante inferencia: {str(e)}")
