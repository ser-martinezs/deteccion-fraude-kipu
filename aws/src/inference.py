#!/usr/bin/env python3
"""
aws/src/inference.py · Lógica de inferencia del endpoint (contenedor de TensorFlow Serving de SageMaker).

Viaja dentro del model.tar.gz en  code/inference.py  (lo copia train_nam.py); el contenedor lo detecta solo y
llama a handler() en cada petición. TensorFlow Serving ejecuta la red; este archivo hace lo demás, solo con
numpy y la biblioteca estándar (el contenedor no trae scikit-learn):

   JSON con variables → preprocesar (igual que el ColumnTransformer de entrenar.py) → TF Serving → calibrar → contrato

ENTRADA  (Content-Type: application/json), cualquiera de estas formas:
   {"variables": {"log_monto_usd": 3.78, "hora_local": 1, ..., "canal": "web", "mcc": "5411"}}     una transacción
   {"log_monto_usd": 3.78, ...}                                                                     una transacción (plano)
   {"instancias": [ {...}, {...} ]}                                                                 varias (evaluación por lotes)
   Las variables son las de datos/tratado/variables.json (15 numéricas + 6 categóricas). Si falta una o viene
   null, se trata como faltante (mediana + indicador), igual que en el entrenamiento. Categorías: texto.

SALIDA   (contrato acordado con servicio-ia)
   una transacción: {"probabilidad": 0.00431, "aportes": {"<factor>": <log-odds>, ...}, "version_modelo": "3.0.0"}
   varias:          {"resultados": [{"probabilidad":..., "aportes":{...}}, ...], "version_modelo": "3.0.0"}
   · probabilidad = probabilidad de fraude EN PRODUCCIÓN (ya calibrada: Platt + corrección de prior).
   · aportes      = contribución de cada factor al log-odds, medida contra la transacción promedio del entrenamiento y
                    en la escala calibrada (positivo = sube el riesgo). Los 3 de mayor |valor| son las "razones".
"""
import json
import math
from pathlib import Path

import numpy as np

_CACHE = {}


# ---------------------------------------------------------------- artefactos
def _dir_modelo():
    aqui = Path(__file__).resolve().parent
    for cand in (aqui.parent, Path("/opt/ml/model")):          # model.tar.gz: <raíz>/code/inference.py y <raíz>/metadata.json
        if (cand / "metadata.json").exists():
            return cand
    raise FileNotFoundError("No encuentro metadata.json junto al modelo (¿el model.tar.gz se armó con train_nam.py?)")


def cargar(base=None):
    """metadata.json + preprocesador.json, una sola vez por proceso (o desde `base`, para pruebas)."""
    clave = str(base) if base else "modelo"
    if clave not in _CACHE:
        raiz = Path(base) if base else _dir_modelo()
        _CACHE[clave] = (json.loads((raiz / "metadata.json").read_text(encoding="utf-8")),
                         json.loads((raiz / "preprocesador.json").read_text(encoding="utf-8")))
    return _CACHE[clave]


# ---------------------------------------------------------------- preprocesamiento (numpy)
def _numero(v, nombre):
    if v is None:
        return math.nan
    if isinstance(v, bool):
        return float(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"variable '{nombre}': se esperaba un número y llegó {v!r}") from None
    return x if math.isfinite(x) else math.nan


def _texto(v, relleno):
    if v is None:
        return relleno
    if isinstance(v, float):
        if math.isnan(v):
            return relleno
        if v.is_integer():
            return str(int(v))                                  # 5411.0 → "5411"
    return str(v)


def transformar(filas, prep):
    """Lista de dicts → matriz float32 (n, n_entrada): imputar mediana + indicador de nulo + estandarizar + one-hot."""
    n = len(filas)
    num = prep["numericas"]
    X = np.array([[_numero(f.get(c), c) for c in num] for f in filas], dtype=np.float64).reshape(n, len(num))
    faltante = np.isnan(X)
    X = np.where(faltante, np.asarray(prep["medianas"], dtype=np.float64), X)
    indicadores = faltante[:, prep["indicadores"]].astype(np.float64) if prep["indicadores"] else np.zeros((n, 0))
    A_num = (np.hstack([X, indicadores]) - np.asarray(prep["media"])) / np.asarray(prep["escala"])
    bloques = [A_num]
    for col in prep["categoricas"]:
        cats = prep["categorias"][col]
        pos = {c: i for i, c in enumerate(cats)}
        M = np.zeros((n, len(cats)))
        for r, f in enumerate(filas):
            j = pos.get(_texto(f.get(col), prep["relleno_cat"]))   # categoría nunca vista → todo ceros (handle_unknown="ignore")
            if j is not None:
                M[r, j] = 1.0
        bloques.append(M)
    return np.hstack(bloques).astype(np.float32)


# ---------------------------------------------------------------- calibración y contrato
def calibrar(p_cruda, cal):
    pc = min(max(float(p_cruda), 1e-9), 1 - 1e-9)
    z = cal["a"] * math.log(pc / (1 - pc)) + cal["b"] + cal["log_razon_prior"]
    return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0


def separar_salidas(pred):
    """Una predicción de TF Serving → (prob cruda, aportes crudos). Elige por forma, no por nombre:
    la de 1 valor es la probabilidad y la de k>1 valores son los aportes (igual que entrenar.salidas_onnx)."""
    vals = [np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in (pred.values() if isinstance(pred, dict) else pred)]
    p = next((v for v in vals if v.size == 1), None)
    c = next((v for v in vals if v.size > 1), None)
    if p is None or c is None:
        raise RuntimeError(f"La salida del modelo no trae probabilidad y aportes (formas: {[v.shape for v in vals]})")
    return float(p[0]), c


def construir_resultados(p_crudas, aportes_crudos, meta):
    cal, base, factores = meta["calibracion"], np.asarray(meta["aportes_base"]), meta["factores"]
    out = []
    for p, c in zip(p_crudas, aportes_crudos):
        if len(c) != len(factores):
            raise RuntimeError(f"El modelo devolvió {len(c)} aportes y metadata.json declara {len(factores)} factores")
        delta = (np.asarray(c) - base) * cal["a"]               # log-odds frente al promedio, en escala calibrada
        out.append({"probabilidad": float(f"{calibrar(p, cal):.8g}"),     # 8 cifras significativas (no decimales)
                    "aportes": {f: round(float(d), 4) for f, d in zip(factores, delta)}})
    return out


# ---------------------------------------------------------------- entrada / handler
def parsear_entrada(texto):
    """Devuelve (lista de dicts de variables, ¿es_lote?)."""
    obj = json.loads(texto)
    if isinstance(obj, dict) and "instancias" in obj:
        filas, lote = obj["instancias"], True
    elif isinstance(obj, dict) and "variables" in obj:
        filas, lote = [obj["variables"]], False
    elif isinstance(obj, dict):
        filas, lote = [obj], False
    elif isinstance(obj, list):
        filas, lote = obj, True
    else:
        raise ValueError("El cuerpo debe ser un objeto JSON con 'variables' o 'instancias'")
    if not filas or not all(isinstance(f, dict) for f in filas):
        raise ValueError("No hay variables que evaluar (se esperaba un objeto o una lista de objetos)")
    return filas, lote


def responder(filas, lote, predicciones, meta):
    p_crudas, aportes = zip(*(separar_salidas(p) for p in predicciones))
    resultados = construir_resultados(p_crudas, aportes, meta)
    version = meta["version_modelo"]
    salida = {"resultados": resultados, "version_modelo": version} if lote else {**resultados[0], "version_modelo": version}
    return json.dumps(salida)


def handler(data, context):
    """Punto de entrada del contenedor de TensorFlow Serving de SageMaker (devuelve (cuerpo, tipo de contenido))."""
    import requests                                              # lo trae el contenedor; se importa aquí para poder probar sin él
    tipo = (getattr(context, "request_content_type", None) or "application/json").split(";")[0].strip()
    if tipo != "application/json":
        raise ValueError(f"Content-Type no soportado: {tipo} (usa application/json)")
    meta, prep = cargar()
    filas, lote = parsear_entrada(data.read().decode("utf-8"))
    X = transformar(filas, prep)
    r = requests.post(context.rest_uri, data=json.dumps({"instances": X.tolist()}),
                      headers={"Content-Type": "application/json"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"TensorFlow Serving respondió {r.status_code}: {r.text[:300]}")
    return responder(filas, lote, r.json()["predictions"], meta), "application/json"
