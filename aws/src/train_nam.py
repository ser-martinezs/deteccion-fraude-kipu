#!/usr/bin/env python3
"""
aws/src/train_nam.py · Entrenamiento de la NAM (red aditiva explicable) DENTRO del contenedor de TensorFlow de SageMaker.

No se ejecuta a mano: lo lanza 02_entrenar_nam.py. Es la misma red que servicio-ia/entrenar.py (entrenar_nam):
una subred pequeña por factor y logit = suma de los aportes. Diferencias, por correr en SageMaker:
  * los datos llegan ya preprocesados (los transformó 01_s3_subir_datos.py con el preprocesador ajustado solo con entrenamiento);
  * depende solo de TensorFlow/Keras y numpy (nada de sklearn/pandas dentro del contenedor);
  * calibra con la VALIDACIÓN (Platt + corrección de prior a producción), igual que entrenar.Calibrador;
  * arma el model.tar.gz que sirve el endpoint:
        1/                    SavedModel con DOS salidas: probabilidad cruda y aportes por factor (TensorFlow Serving)
        code/inference.py     handler del endpoint (contrato probabilidad / aportes / version_modelo)
        metadata.json         version_modelo, factores, aportes_base, calibración
        preprocesador.json    el preprocesador de entrenamiento en JSON (lo aplica inference.py con numpy)

Canales (SageMaker los monta en /opt/ml/input/data/<canal>):
  train/train.csv · validation/validation.csv · config/grupos.json + config/preprocesador.json
  (columnas de los CSV: y, etiquetada, x000, x001, ...)
Hiperparámetros: ver parse_args().  Métricas para CloudWatch: líneas  "val_pr_auc_prod=0.123456".
"""
import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---- cifras del caso (iguales a servicio-ia/entrenar.py): ponderan la validación a la proporción de producción
TX_DIA = 200_000
PI_PROD = 0.00012
FRAUDES_DIA = TX_DIA * PI_PROD
LEGITIMAS_DIA = TX_DIA - FRAUDES_DIA


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="binary:logistic", help="solo binary:logistic (casos 1 y 3)")
    ap.add_argument("--version", default="3.0.0", help="version_modelo que devolverá el endpoint")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=3e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--capas", default="16,8", help="neuronas de cada capa oculta de cada subred, separadas por coma")
    ap.add_argument("--peso_fraude", type=float, default=20.0)
    ap.add_argument("--peso_no_revisadas", type=float, default=1.0)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--eval_cada", type=int, default=10, help="cada cuántas épocas se mide en validación")
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    ap.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    ap.add_argument("--config", default=os.environ.get("SM_CHANNEL_CONFIG", "/opt/ml/input/data/config"))
    ap.add_argument("--salida_modelo", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    ap.add_argument("--salida_datos", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    args, _desconocidos = ap.parse_known_args(argv)      # SageMaker agrega argumentos propios: se ignoran
    return args


# ---------------------------------------------------------------- datos, métrica y calibración (solo numpy)
def cargar_csv(carpeta, nombre):
    M = np.loadtxt(Path(carpeta) / nombre, delimiter=",", skiprows=1, dtype=np.float32)
    return M[:, 0], M[:, 1], M[:, 2:]            # y, etiquetada, X


def pesos_dia(y):
    y = np.asarray(y)
    return np.where(y == 1, FRAUDES_DIA / (y == 1).sum(), LEGITIMAS_DIA / (y == 0).sum())


def pr_auc_ponderado(y, p, w):
    """Average precision con pesos: igual a sklearn.average_precision_score(y, p, sample_weight=w)."""
    y, p, w = (np.asarray(a, dtype=np.float64) for a in (y, p, w))
    orden = np.argsort(-p, kind="mergesort")
    y, p, w = y[orden], p[orden], w[orden]
    ultimos = np.r_[np.where(np.diff(p))[0], len(p) - 1]         # un punto por umbral distinto
    tp = np.cumsum(w * y)[ultimos]
    fp = np.cumsum(w * (1 - y))[ultimos]
    precision, recall = tp / (tp + fp), tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def ajustar_platt(p, y, c_inv=1e6, iteraciones=100):
    """a·logit(p) + b por máxima verosimilitud (Newton): equivale a LogisticRegression(C=1e6) de entrenar.Calibrador."""
    z = logit(p)
    X = np.column_stack([z, np.ones_like(z)])
    y = np.asarray(y, dtype=np.float64)
    theta = np.array([1.0, 0.0])
    reg = np.diag([1.0 / c_inv, 0.0])                            # penalización L2 mínima sobre la pendiente
    for _ in range(iteraciones):
        q = 1.0 / (1.0 + np.exp(-(X @ theta)))
        grad = X.T @ (q - y) + reg @ theta
        hess = (X * (q * (1 - q))[:, None]).T @ X + reg + 1e-12 * np.eye(2)
        paso = np.linalg.solve(hess, grad)
        theta = theta - paso
        if np.abs(paso).max() < 1e-10:
            break
    return float(theta[0]), float(theta[1])


def calibracion(p_val, y_val):
    a, b = ajustar_platt(p_val, y_val)
    pi_ref = float(np.mean(y_val))
    return {"a": a, "b": b, "log_razon_prior": math.log(PI_PROD / (1 - PI_PROD)) - math.log(pi_ref / (1 - pi_ref)),
            "pi_muestra_validacion": pi_ref, "pi_produccion": PI_PROD}


# ---------------------------------------------------------------- la red
def construir_nam(grupos, n_in, capas, l2, learning_rate):
    """Devuelve (modelo, explicable): comparten capas. `modelo` se entrena (probabilidad);
    `explicable` tiene dos salidas {prob, aportes} y es el que se exporta al endpoint."""
    import keras
    entrada = keras.Input(shape=(n_in,), name="x")
    salidas = []
    for nombre, idx in grupos.items():
        sel = np.zeros((n_in, len(idx)), dtype="float32")             # matriz 0/1 que recorta las columnas del factor
        for k, j in enumerate(idx):
            sel[j, k] = 1.0
        capa_sel = keras.layers.Dense(len(idx), use_bias=False, trainable=False, name=f"sel_{nombre}")
        h = capa_sel(entrada)
        capa_sel.set_weights([sel])
        for i, u in enumerate(capas):
            h = keras.layers.Dense(u, activation="relu", kernel_regularizer=keras.regularizers.l2(l2), name=f"f_{nombre}_{i}")(h)
        salidas.append(keras.layers.Dense(1, name=f"aporte_{nombre}")(h))
    contribs = keras.layers.Concatenate(name="aportes")(salidas)
    capa_suma = keras.layers.Dense(1, use_bias=False, trainable=False, name="suma")
    logit_ = capa_suma(contribs)
    capa_suma.set_weights([np.ones((len(grupos), 1), dtype="float32")])        # logit = Σ aportes
    prob = keras.layers.Activation("sigmoid", name="prob")(logit_)
    modelo = keras.Model(entrada, prob)
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="binary_crossentropy")
    explicable = keras.Model(entrada, {"prob": prob, "aportes": contribs}, name="nam_explicable")
    return modelo, explicable


def _por_forma(salidas):
    """dict o lista de salidas → (probabilidad (n,), aportes (n,k)); elige por forma, no por nombre."""
    vals = [np.asarray(v) for v in (salidas.values() if isinstance(salidas, dict) else salidas)]
    p = next(v for v in vals if v.ndim == 2 and v.shape[1] == 1)
    c = next(v for v in vals if v.ndim == 2 and v.shape[1] > 1)
    return p.ravel(), c


def predecir_explicable(explicable, A):
    return _por_forma(explicable.predict(np.asarray(A, dtype="float32"), verbose=0, batch_size=4096))


def exportar_saved_model(explicable, destino):
    """SavedModel con firma serving_default (lo que carga TensorFlow Serving). Keras 3: export(); Keras 2: saved_model.save."""
    import tensorflow as tf
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(explicable, "export"):
        explicable.export(str(destino))
    else:
        tf.saved_model.save(explicable, str(destino))


def verificar_saved_model(destino, explicable, A):
    """El SavedModel recargado debe dar las mismas probabilidades y aportes que la red en memoria."""
    import tensorflow as tf
    try:
        firma = tf.saved_model.load(str(destino)).signatures["serving_default"]
        nombre_in = list(firma.structured_input_signature[1].keys())[0]
        muestra = np.asarray(A[:64], dtype="float32")
        salidas = firma(**{nombre_in: tf.constant(muestra)})
        print(f"firma serving_default: entrada '{nombre_in}' · salidas {list(salidas.keys())}")
        p_saved, c_saved = _por_forma({k: v.numpy() for k, v in salidas.items()})
    except Exception as e:                       # la verificación es una red de seguridad: si no se puede ejecutar, se avisa
        print(f"AVISO: no se pudo verificar el SavedModel ({type(e).__name__}: {str(e)[:200]})")
        return None
    p_ref, c_ref = predecir_explicable(explicable, muestra)
    dif_p, dif_c = float(np.abs(p_saved - p_ref).max()), float(np.abs(c_saved - c_ref).max())
    if dif_p > 1e-4 or dif_c > 1e-3:
        raise RuntimeError(f"El SavedModel no reproduce a la red (prob {dif_p:.1e}, aportes {dif_c:.1e})")
    return dif_p, dif_c


# ---------------------------------------------------------------- programa principal
def main(argv=None):
    args = parse_args(argv)
    if args.objective != "binary:logistic":
        sys.exit(f"ERROR: objective={args.objective}. Esta NAM es binaria (casos 1 y 3: binary:logistic).")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(args.semilla)

    y_tr, etiq_tr, A_tr = cargar_csv(args.train, "train.csv")
    y_va, _, A_va = cargar_csv(args.validation, "validation.csv")
    grupos = json.loads((Path(args.config) / "grupos.json").read_text())
    prep = json.loads((Path(args.config) / "preprocesador.json").read_text())
    n_in = A_tr.shape[1]
    esperado = len(prep["numericas"]) + len(prep["indicadores"]) + sum(len(v) for v in prep["categorias"].values())
    if esperado != n_in:
        sys.exit(f"ERROR: preprocesador.json produce {esperado} columnas y los CSV traen {n_in}: se generaron con preprocesadores distintos")
    capas = [int(c) for c in str(args.capas).split(",") if c.strip()]
    print(f"TensorFlow {tf.__version__} · Keras {keras.__version__} · entrada {n_in} columnas · {len(grupos)} factores · capas {capas}")
    print(f"entrenamiento {len(y_tr):,} filas ({int(y_tr.sum())} fraudes) · validación {len(y_va):,} filas ({int(y_va.sum())} fraudes)")

    w_tr = np.where(y_tr == 1, args.peso_fraude, np.where(etiq_tr == 1, 1.0, args.peso_no_revisadas)).astype("float32")
    wd_va = pesos_dia(y_va)

    modelo, explicable = construir_nam(grupos, n_in, capas, args.l2, args.learning_rate)
    mejor_ap, mejores_pesos, mejor_epoca, hechas = -1.0, None, 0, 0
    while hechas < args.epochs:                  # de a eval_cada épocas; se conserva la mejor según VALIDACIÓN
        tramo = min(args.eval_cada, args.epochs - hechas)
        modelo.fit(A_tr, y_tr, sample_weight=w_tr, epochs=tramo, batch_size=args.batch_size, verbose=0)
        hechas += tramo
        ap = pr_auc_ponderado(y_va, modelo.predict(A_va, verbose=0, batch_size=4096).ravel(), wd_va)
        print(f"epoca={hechas} val_pr_auc_prod={ap:.6f}", flush=True)
        if ap > mejor_ap:
            mejor_ap, mejores_pesos, mejor_epoca = ap, modelo.get_weights(), hechas
    modelo.set_weights(mejores_pesos)
    print(f"mejor_epoca={mejor_epoca} mejor_val_pr_auc_prod={mejor_ap:.6f}", flush=True)

    # ---- calibración con VALIDACIÓN y aporte promedio de entrenamiento (referencia de las "razones")
    p_val, _ = predecir_explicable(explicable, A_va)
    cal = calibracion(p_val, y_va)
    _, aportes_tr = predecir_explicable(explicable, A_tr)
    print(f"calibración: a={cal['a']:.4f} b={cal['b']:.4f} log_razon_prior={cal['log_razon_prior']:.4f}")

    # ---- model.tar.gz: 1/ + code/inference.py + metadata.json + preprocesador.json
    raiz = Path(args.salida_modelo)
    destino = raiz / "1"
    exportar_saved_model(explicable, destino)
    ver = verificar_saved_model(destino, explicable, A_va)
    print(f"SavedModel exportado en {destino}" + (f" · verificado (dif. prob {ver[0]:.1e}, aportes {ver[1]:.1e})" if ver else ""))

    (raiz / "code").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("inference.py"), raiz / "code" / "inference.py")
    shutil.copy(Path(__file__).with_name("requirements.txt"), raiz / "code" / "requirements.txt")
    shutil.copy(Path(args.config) / "preprocesador.json", raiz / "preprocesador.json")
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "nombre": "fraude-kipu-nam", "version_modelo": args.version, "formato": "saved_model (TensorFlow Serving)",
        "framework": f"tensorflow {tf.__version__}", "entrenado_en": ahora, "trabajo_sagemaker": os.environ.get("TRAINING_JOB_NAME"),
        "arquitectura": "NAM: una subred por factor, logit = suma de aportes (explicable por construcción)",
        "factores": list(grupos), "aportes_base": [float(v) for v in aportes_tr.mean(axis=0)], "calibracion": cal,
        "n_entrada": n_in, "mejor_epoca": mejor_epoca, "mejor_val_pr_auc_prod": mejor_ap,
        "filas_entrenamiento": int(len(y_tr)), "fraudes_entrenamiento": int(y_tr.sum())}
    (raiz / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    Path(args.salida_datos).mkdir(parents=True, exist_ok=True)
    (Path(args.salida_datos) / "resumen_entrenamiento.json").write_text(json.dumps({
        **{k: metadata[k] for k in ("version_modelo", "mejor_epoca", "mejor_val_pr_auc_prod", "n_entrada", "calibracion")},
        "n_factores": len(grupos),
        "hiperparametros": {k: v for k, v in vars(args).items() if k in
                            ("objective", "version", "epochs", "learning_rate", "batch_size", "l2", "capas", "peso_fraude", "peso_no_revisadas", "semilla")},
        "tensorflow": tf.__version__, "keras": keras.__version__}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
