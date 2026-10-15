#!/usr/bin/env python3
"""
servicio-ia/entrenar.py · Entrena el modelo de fraude, lo mide sobre PRUEBA y aplica la puerta de calidad.

Lo ejecuta el job de la imagen de IA DESPUÉS de datos/preparar.py:

    python datos/preparar.py  --entrada datos/crudo --salida datos/tratado
    python servicio-ia/entrenar.py --tratado datos/tratado --salida servicio-ia/model   # falla (código 1) si la métrica baja del umbral
    docker build ...                                                                    # solo corre si el paso anterior terminó en 0

Qué hace:
  - Lee variables_modelo.csv (la separación entrenamiento/validación/prueba ya viene hecha por preparar.py).
  - Ajusta el preprocesador SOLO con entrenamiento y entrena la red aditiva explicable (TensorFlow/Keras).
  - Calibra con validación y propone umbrales (los fija riesgo: quedan en umbrales.json, separados del modelo).
  - Informa la métrica sobre PRUEBA y la compara con servicio-ia/calidad.json (umbral fijado por el equipo).
  - Solo si la métrica cumple: exporta modelo.onnx + preprocesador.joblib + metadata.json + umbrales.json.

Códigos de salida: 0 = ok y modelo exportado · 1 = la métrica no alcanza el umbral (no se exporta nada)
                   2 = datos inválidos o faltan archivos · 3 = TensorFlow no está disponible.
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
SEMILLA = 42

# ---------- Cifras del caso (secciones 2 y 8) ----------
TX_DIA = 200_000
PI_PROD = 0.00012                                 # 0,012% de fraude en producción (la muestra tiene ~0,5%)
FRAUDES_DIA = TX_DIA * PI_PROD
LEGITIMAS_DIA = TX_DIA - FRAUDES_DIA
COSTO_FP_BLOQUEO = 1_162_600 / (460 * 365)        # USD por cliente bueno bloqueado ≈ 6,92
CAPACIDAD_REVISION_DIA = 4_800
TURNO = (8, 22)
FP_OBJETIVO = 0.15
# ---------- SUPUESTOS (el caso no los entrega) ----------
COSTO_REVISION = 0.10                             # USD por revisión manual
EFECTIVIDAD_REVISION = 0.90                       # fracción de fraudes revisados en turno que el analista confirma
PESO_FRAUDE = 20.0                                # peso de la clase fraude al entrenar (se recalibra después)
PESO_NO_REVISADAS = 1.0                           # las filas sin etiqueta entran como legítimas con este peso


class ErrorDeDatos(Exception):
    pass


# ================================================================== métricas de negocio a proporción de producción
def pesos_dia(y):
    """Peso de cada fila en 'transacciones por día de producción': el fraude suma ~24/día; las legítimas, ~199.976/día."""
    y = np.asarray(y)
    return np.where(y == 1, FRAUDES_DIA / (y == 1).sum(), LEGITIMAS_DIA / (y == 0).sum())


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


class Calibrador:
    """Platt (a·logit + b) ajustado en validación + corrección de prior muestra -> producción."""
    def ajustar(self, p_val, y_val):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_val).reshape(-1, 1), y_val)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        self.pi_ref = float(np.mean(y_val))
        self.log_razon_prior = float(np.log(PI_PROD / (1 - PI_PROD)) - np.log(self.pi_ref / (1 - self.pi_ref)))
        return self

    def __call__(self, p):
        return 1 / (1 + np.exp(-(self.a * logit(p) + self.b + self.log_razon_prior)))

    def a_dict(self):
        return {"a": self.a, "b": self.b, "log_razon_prior": self.log_razon_prior,
                "pi_muestra_validacion": self.pi_ref, "pi_produccion": PI_PROD}


def decidir_zona(p, hora, umb, dura=None):
    """0 aprobar · 1 revisar · 2 bloquear. De madrugada no hay zona 'revisar' (no hay analistas: se revisaría con el dinero liquidado)."""
    p, hora = np.asarray(p), np.asarray(hora)
    turno = (hora >= TURNO[0]) & (hora < TURNO[1])
    ub = np.where(turno, umb["bloquear_dia"], umb["bloquear_noche"])
    ur = np.where(turno, umb["revisar_dia"], np.inf)
    z = np.where(p >= ub, 2, np.where(p >= ur, 1, 0))
    if dura is not None:
        z = np.where(np.asarray(dura), 2, z)           # la regla dura manda sobre el modelo
    return z


def proyectar(y, z, usd, wd):
    y, z, usd, wd = (np.asarray(a) for a in (y, z, usd, wd))
    f, l = y == 1, y == 0
    bloq_f, bloq_l = wd[f & (z == 2)].sum(), wd[l & (z == 2)].sum()
    rev_f, rev_l = wd[f & (z == 1)].sum(), wd[l & (z == 1)].sum()
    perdida = 365 * ((wd * usd)[f & (z == 0)].sum() + (1 - EFECTIVIDAD_REVISION) * (wd * usd)[f & (z == 1)].sum())
    costo = perdida + 365 * COSTO_FP_BLOQUEO * bloq_l + 365 * COSTO_REVISION * (rev_f + rev_l)
    total_f = wd[f].sum()
    return {"bloqueos_dia": bloq_f + bloq_l, "fp_dia": bloq_l,
            "fp_sobre_bloqueos": bloq_l / (bloq_f + bloq_l) if (bloq_f + bloq_l) > 0 else 0.0,
            "revisiones_dia": rev_f + rev_l, "deteccion": (bloq_f + EFECTIVIDAD_REVISION * rev_f) / total_f if total_f > 0 else float("nan"),
            "fraude_no_detectado_usd": perdida, "costo_usd": costo}


GRILLA = 1 / (1 + np.exp(-np.arange(-14, 6.01, 0.25)))


def _elegir(candidatos):
    return candidatos[len(candidatos) // 2]            # entre empates de costo, el centro de la meseta


def buscar_umbrales(p, y, usd, hora, dura):
    """Mínimo costo anual en validación, sujeto a revisiones ≤ capacidad y FP sobre bloqueos ≤ 15%.
    Las filas que decide la regla dura no participan: la regla manda y el modelo no puede cambiarlas."""
    p, y, usd, hora, dura = (np.asarray(a) for a in (p, y, usd, hora, dura))
    wd = pesos_dia(y)
    m = ~dura
    p, y, usd, hora, wd = p[m], y[m], usd[m], hora[m], wd[m]
    turno = (hora >= TURNO[0]) & (hora < TURNO[1])
    info = {}
    pn, yn, un, wn = p[~turno], y[~turno], usd[~turno], wd[~turno]
    cand = []
    for ub in GRILLA:
        r = proyectar(yn, np.where(pn >= ub, 2, 0), un, wn)
        cand.append((r["costo_usd"], ub, r["fp_sobre_bloqueos"]))
    ok = [c for c in cand if c[2] <= FP_OBJETIVO] or [max(cand, key=lambda c: c[1])]
    mejor = min(c[0] for c in ok)
    bloq_noche = _elegir([c[1] for c in ok if c[0] <= mejor + 1e-6])
    info["noche_cumple_fp"] = any(c[2] <= FP_OBJETIVO for c in cand)
    pd_, yd, ud, wdd = p[turno], y[turno], usd[turno], wd[turno]
    cand = []
    for ur in GRILLA:
        for ub in GRILLA[GRILLA >= ur]:
            z = np.where(pd_ >= ub, 2, np.where(pd_ >= ur, 1, 0))
            r = proyectar(yd, z, ud, wdd)
            cand.append((r["costo_usd"], ur, ub, r["fp_sobre_bloqueos"], r["revisiones_dia"]))
    ok = [c for c in cand if c[3] <= FP_OBJETIVO and c[4] <= CAPACIDAD_REVISION_DIA]
    info["dia_cumple_restricciones"] = bool(ok)
    if not ok:
        ok = [c for c in cand if c[4] <= CAPACIDAD_REVISION_DIA]
        ok = [min(ok, key=lambda c: (c[3], c[0]))] if ok else [max(cand, key=lambda c: c[2])]
    mejor = min(c[0] for c in ok)
    elegido = _elegir([c for c in ok if c[0] <= mejor + 1e-6])
    return {"revisar_dia": float(elegido[1]), "bloquear_dia": float(elegido[2]), "bloquear_noche": float(bloq_noche)}, info


# ================================================================== preprocesador y factores
def construir_preprocesador(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)),   # nulo -> mediana + indicador "faltaba"
                          ("escalar", StandardScaler())]), num),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="constant", fill_value="desconocido")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def grupos_de(nombres, cat):
    """Cada columna de la matriz pertenece a un 'factor' del negocio: la variable, su indicador de nulo o su one-hot."""
    cats = sorted(cat, key=len, reverse=True)
    grupos = {}
    for j, nom in enumerate(nombres):
        if nom.startswith("num__"):
            base = nom[5:]
            base = base.split("missingindicator_", 1)[1] if base.startswith("missingindicator_") else base
        else:
            base = next(c for c in cats if nom.startswith(f"cat__{c}_"))
        grupos.setdefault(base, []).append(j)
    return grupos


# ================================================================== la red aditiva explicable (TensorFlow / Keras)
def entrenar_nam(A_train, y_train, w_train, A_val, y_val, wd_val, grupos, capas=(16, 8), l2=1e-4, epocas=100, lr=3e-3, batch=256):
    """NAM: una subred pequeña por factor; logit = suma de los aportes. El aporte de cada factor es EXACTO (RT8, RF2)."""
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    try:
        import keras
        import tensorflow as tf
    except ImportError as e:
        raise SystemExit(f"TensorFlow/Keras no está instalado ({e}). Instala servicio-ia/requirements-ci.txt") from None

    keras.utils.set_random_seed(SEMILLA)
    n_in = A_train.shape[1]
    entrada = keras.Input(shape=(n_in,), name="x")
    salidas = []
    for nombre, idx in grupos.items():
        sel = np.zeros((n_in, len(idx)), dtype="float32")             # matriz 0/1 que 'recorta' las columnas del factor
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
    capa_suma.set_weights([np.ones((len(grupos), 1), dtype="float32")])     # logit = Σ aportes
    prob = keras.layers.Activation("sigmoid", name="prob")(logit_)
    modelo = keras.Model(entrada, prob)
    modelo.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    expl = keras.Model(entrada, [prob, contribs])

    def predecir(A):
        return modelo.predict(np.asarray(A, dtype="float32"), verbose=0, batch_size=4096).ravel()

    def aportes(A):
        return np.asarray(expl.predict(np.asarray(A, dtype="float32"), verbose=0, batch_size=4096)[1])

    y_tr = np.asarray(y_train, dtype="float32")
    mejor_ap, mejores_pesos, hechas = -1.0, None, 0
    while hechas < epocas:                                           # de a 10 épocas; nos quedamos con la mejor según VALIDACIÓN
        tramo = min(10, epocas - hechas)
        modelo.fit(A_train, y_tr, sample_weight=w_train, epochs=tramo, batch_size=batch, verbose=0)
        hechas += tramo
        ap = average_precision_score(y_val, predecir(A_val), sample_weight=wd_val)
        print(f"   época {hechas:>3} · PR-AUC (producción) en validación {ap:.3f}")
        if ap > mejor_ap:
            mejor_ap, mejores_pesos = ap, modelo.get_weights()
    modelo.set_weights(mejores_pesos)

    def exportar_onnx(ruta):
        try:
            expl.export(str(ruta), format="onnx")                    # Keras 3 usa tf2onnx por debajo
        except Exception as e:
            print("   export(format='onnx') falló:", repr(e)[:120], "→ intento con tf2onnx")
            import tf2onnx
            spec = (tf.TensorSpec((None, n_in), tf.float32, name="x"),)
            tf2onnx.convert.from_keras(expl, input_signature=spec, opset=17, output_path=str(ruta))

    return {"predecir": predecir, "aportes": aportes, "exportar_onnx": exportar_onnx, "mejor_pr_auc_val": mejor_ap,
            "framework": f"tensorflow {tf.__version__}"}


# ================================================================== exportación y verificación
def salidas_onnx(sesion, A):
    nombre = sesion.get_inputs()[0].name
    outs = sesion.run(None, {nombre: np.asarray(A, dtype=np.float32)})
    p = next(o for o in outs if o.ndim == 2 and o.shape[1] == 1).ravel()
    c = next((o for o in outs if o.ndim == 2 and o.shape[1] > 1), None)
    return p, c


def verificar_onnx(ruta_onnx, red, A):
    """onnxruntime (lo que usará la API) debe reproducir probabilidad y aportes de la red de Keras."""
    import onnxruntime as ort
    sesion = ort.InferenceSession(str(ruta_onnx), providers=["CPUExecutionProvider"])
    p_onnx, c_onnx = salidas_onnx(sesion, A)
    dif_p = float(np.abs(p_onnx - red["predecir"](A)).max())
    dif_c = float(np.abs(c_onnx - red["aportes"](A)).max())
    if not (dif_p < 1e-4 and dif_c < 1e-3):
        raise RuntimeError(f"El ONNX no reproduce a la red (prob {dif_p:.1e}, aportes {dif_c:.1e})")
    return dif_p, dif_c


def a_python(o):
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, dict):
        return {k: a_python(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [a_python(v) for v in o]
    return o


# ================================================================== programa principal
def main(argv=None):
    ap = argparse.ArgumentParser(description="Entrena el modelo de fraude y aplica la puerta de calidad")
    ap.add_argument("--tratado", type=Path, default=RAIZ / "datos" / "tratado")
    ap.add_argument("--salida", type=Path, default=RAIZ / "servicio-ia" / "model")
    ap.add_argument("--calidad", type=Path, default=RAIZ / "servicio-ia" / "calidad.json")
    ap.add_argument("--informe", type=Path, default=RAIZ / "servicio-ia" / "reporte_entrenamiento.json")
    ap.add_argument("--version", default="2.0.0")
    ap.add_argument("--epocas", type=int, default=100)
    args = ap.parse_args(argv)
    np.random.seed(SEMILLA)
    t0 = time.perf_counter()

    # ---------- 1. datos ya tratados y ya separados por preparar.py
    try:
        var = json.loads((args.tratado / "variables.json").read_text())
        d = pd.read_csv(args.tratado / "variables_modelo.csv", parse_dates=["ts_utc"], dtype={c: str for c in var["categoricas"]})
        manifiesto = json.loads((args.tratado / "manifiesto.json").read_text())
        calidad = json.loads(args.calidad.read_text())
    except FileNotFoundError as e:
        print(f"ERROR: falta {e.filename}. ¿Corriste datos/preparar.py antes?", file=sys.stderr)
        return 2
    NUM, CAT = var["numericas"], var["categoricas"]
    FEATURES = NUM + CAT
    for c in CAT:                                                 # categorías como texto (igual que las recibirá la API); el nulo es NaN
        d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
    tr, va, te = (d[d["split"] == s].copy() for s in ["train", "val", "test"])
    if min(len(tr), len(va), len(te)) == 0:
        print("ERROR: algún conjunto quedó vacío", file=sys.stderr)
        return 2
    y_train, y_val, y_test = (x["y"].to_numpy() for x in (tr, va, te))
    wd_val, wd_test = pesos_dia(y_val), pesos_dia(y_test)
    usd_val, usd_test = va["monto_usd"].to_numpy(), te["monto_usd"].to_numpy()
    hora_val, hora_test = va["hora_local"].to_numpy(), te["hora_local"].to_numpy()
    dura_val, dura_test = va["lista_negra_dura"].astype(bool).to_numpy(), te["lista_negra_dura"].astype(bool).to_numpy()
    print(f"Entrenamiento {len(tr):,} · validación {len(va):,} · prueba {len(te):,} filas ({int(y_test.sum())} fraudes en prueba)")

    # ---------- 2. preprocesador (SOLO con entrenamiento) y matrices
    prep = construir_preprocesador(NUM, CAT).fit(tr[FEATURES])
    A_train, A_val, A_test = (prep.transform(x[FEATURES]).astype(np.float32) for x in (tr, va, te))
    grupos = grupos_de(list(prep.get_feature_names_out()), CAT)
    w_train = np.where(tr["y"] == 1, PESO_FRAUDE, np.where(tr["etiquetada"], 1.0, PESO_NO_REVISADAS)).astype(np.float32)

    def evaluar(p_val, p_test):
        cal = Calibrador().ajustar(p_val, y_val)
        umb, info = buscar_umbrales(cal(p_val), y_val, usd_val, hora_val, dura_val)
        pt = cal(p_test)
        m = ~dura_test
        r = proyectar(y_test[m], decidir_zona(pt, hora_test, umb)[m], usd_test[m], wd_test[m])
        met = {"roc_auc": float(roc_auc_score(y_test, p_test)),
               "pr_auc_prod": float(average_precision_score(y_test, p_test, sample_weight=wd_test)),
               "pr_auc_muestra": float(average_precision_score(y_test, p_test)),
               "costo_anual_usd": r["costo_usd"], "fraude_no_detectado_usd": r["fraude_no_detectado_usd"],
               "fp_sobre_bloqueos": r["fp_sobre_bloqueos"], "bloqueos_dia": r["bloqueos_dia"], "revisiones_dia": r["revisiones_dia"],
               "deteccion": r["deteccion"], "n_fraudes_prueba": int(y_test.sum()), "n_filas_prueba": int(len(y_test))}
        return met, cal, umb, info

    # ---------- 3. referencia: regresión logística (la vara que la red debe respetar)
    lr = LogisticRegression(max_iter=3000).fit(A_train, y_train, sample_weight=w_train)
    met_lr, _, _, _ = evaluar(lr.predict_proba(A_val)[:, 1], lr.predict_proba(A_test)[:, 1])

    # ---------- 4. la red aditiva explicable
    print("Entrenando la red aditiva (TensorFlow)...")
    red = entrenar_nam(A_train, y_train, w_train, A_val, y_val, wd_val, grupos, epocas=args.epocas)
    met, cal, umb, info = evaluar(red["predecir"](A_val), red["predecir"](A_test))

    # ---------- 5. puerta de calidad: umbral fijado por el equipo (calidad.json), medido sobre PRUEBA
    minimos = {"roc_auc": calidad["roc_auc_min"], "pr_auc_prod": calidad["pr_auc_prod_min"]}
    fallas = [f"{k} = {met[k]:.3f} < mínimo {v}" for k, v in minimos.items() if not met[k] >= v]
    if met["n_fraudes_prueba"] < calidad.get("fraudes_prueba_min", 0):
        fallas.append(f"solo {met['n_fraudes_prueba']} fraudes en prueba (mínimo {calidad['fraudes_prueba_min']}): la métrica no es confiable")
    aprobado = not fallas

    print("\n  métrica sobre PRUEBA            red aditiva    LogReg (ref.)   mínimo del equipo")
    for k in ["roc_auc", "pr_auc_prod"]:
        print(f"  {k:<30} {met[k]:>10.3f} {met_lr[k]:>14.3f} {minimos[k]:>16}")
    print(f"  costo anual proyectado (USD)   {met['costo_anual_usd']:>10,.0f} {met_lr['costo_anual_usd']:>14,.0f}")
    print(f"  ({met['n_fraudes_prueba']} fraudes en prueba: diferencias pequeñas no son diferencias)")

    informe = {"generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"), "version": args.version, "framework": red["framework"],
               "metricas_prueba": met, "referencia_regresion_logistica": met_lr, "minimos_equipo": minimos, "aprobado": aprobado, "fallas": fallas,
               "umbrales_propuestos": umb, "umbrales_cumplen_restricciones_en_validacion": info, "segundos": round(time.perf_counter() - t0, 1)}
    args.informe.parent.mkdir(parents=True, exist_ok=True)
    args.informe.write_text(json.dumps(a_python(informe), indent=2, ensure_ascii=False))

    if not aprobado:
        print("\nPUERTA DE CALIDAD: NO SE EXPORTA EL MODELO →", "; ".join(fallas), file=sys.stderr)
        return 1
    print("\nPUERTA DE CALIDAD: aprobada")

    # ---------- 6. exportar (a una carpeta temporal; recién al final reemplaza a la vigente)
    tmp = args.salida.with_name(args.salida.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    joblib.dump(prep, tmp / "preprocesador.joblib")
    red["exportar_onnx"](tmp / "modelo.onnx")
    dif_p, dif_c = verificar_onnx(tmp / "modelo.onnx", red, A_test)
    print(f"ONNX verificado: diferencia máxima en probabilidad {dif_p:.1e} y en aportes {dif_c:.1e}")
    nombres = list(grupos)
    metadata = {
        "nombre": "fraude-kipu", "version": args.version, "formato": "onnx", "framework": red["framework"],
        "entrenado_en": informe["generado_en"], "sklearn_version": sklearn.__version__,
        "arquitectura": "NAM: una subred por factor, logit = suma de aportes (explicable por construcción)",
        "variables_numericas": NUM, "variables_categoricas": CAT, "factores": nombres,
        "aportes_base": [float(v) for v in red["aportes"](A_train).mean(axis=0)],
        "calibracion": cal.a_dict(),
        "zona_horaria": var["parametros"],
        "supuestos": {"costo_revision_usd": COSTO_REVISION, "efectividad_revision": EFECTIVIDAD_REVISION, "costo_fp_bloqueo_usd": round(COSTO_FP_BLOQUEO, 2),
                      "peso_no_revisadas": PESO_NO_REVISADAS, "peso_fraude_entrenamiento": PESO_FRAUDE},
        "datos": {"manifiesto_preparar": manifiesto, "filas_entrenamiento": int(len(tr)), "fraudes_entrenamiento": int(y_train.sum()),
                  "periodo_entrenamiento": [str(tr["ts_utc"].min()), str(tr["ts_utc"].max())],
                  "periodo_prueba": [str(te["ts_utc"].min()), str(te["ts_utc"].max())]},
        "puerta_de_calidad": {"minimos_equipo": minimos, "resultado": "aprobada"},
        "metricas_prueba": met, "referencia_regresion_logistica": met_lr,
    }
    (tmp / "metadata.json").write_text(json.dumps(a_python(metadata), indent=2, ensure_ascii=False))
    umbrales = {"version": "u-0001", "version_modelo": args.version, "umbrales": umb,
                "unidad": "probabilidad de fraude EN PRODUCCIÓN (ya calibrada)", "reglas_duras": ["lista_negra:solicitud_cumplimiento"],
                "creado_por": "servicio-ia/entrenar.py (propuesta)", "creado_en": informe["generado_en"],
                "motivo": "mínimo costo en validación con revisiones ≤ capacidad de analistas y FP ≤ 15%",
                "autorizado_por": "PENDIENTE: gerencia de riesgo (DA5 / RO5)"}
    (tmp / "umbrales.json").write_text(json.dumps(a_python(umbrales), indent=2, ensure_ascii=False))
    shutil.rmtree(args.salida, ignore_errors=True)
    tmp.rename(args.salida)
    print(f"Modelo exportado en {args.salida} ({', '.join(sorted(p.name for p in args.salida.iterdir()))}) · {informe['segundos']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
