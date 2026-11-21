#!/usr/bin/env python3
"""
04_invocar_y_metricas.py · Invoca el endpoint con la PRUEBA y calcula las métricas del caso 3 (fraude).
                           →  consola: CloudWatch → Metrics → SageMaker → Endpoint (y namespace ITY1102/EP3)

El endpoint recibe las variables en JSON y devuelve {probabilidad, aportes, version_modelo} (ver src/inference.py):
la probabilidad ya viene calibrada a producción (la calibración se ajustó con la validación dentro del entrenamiento).

Qué hace:
  1. Invoca el endpoint con la validación → busca los umbrales de costo mínimo, como servicio-ia/entrenar.py
     (la prueba no participa en esa decisión).
  2. Invoca el endpoint con la prueba (que nunca se subió a S3) → ROC-AUC, PR-AUC (re-ponderada a producción), costo
     anual proyectado, fraude no detectado, FP sobre bloqueos, bloqueos y revisiones por día y detección.
  3. Compara con los mínimos del equipo (servicio-ia/calidad.json) y con el entrenamiento local (reporte_entrenamiento.json).
  4. Hace UNA invocación individual y la muestra completa (el contrato que consume servicio-ia).
  5. Publica las métricas en CloudWatch (namespace ITY1102/EP3) y lee las del endpoint (Invocations, ModelLatency,
     ModelSetupTime, errores 4XX/5XX). CloudWatch demora 1-3 min en mostrarlas.
  Guarda aws/metricas_prueba_sagemaker.json y aws/umbrales_sagemaker.json (mismo formato que servicio-ia/model/umbrales.json).

Uso:  python aws/04_invocar_y_metricas.py [--lote 200] [--sin-cloudwatch] [--espera-cw 240]
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import config

NAMESPACE_PROPIO = "ITY1102/EP3"


# ---------------------------------------------------------------- invocación
def a_instancias(df, NUM, CAT):
    """DataFrame → lista de dicts JSON con las variables del modelo (NaN → null; categorías como texto)."""
    nulo = lambda v: v is None or (isinstance(v, float) and np.isnan(v))
    return [{**{c: (None if nulo(r[c]) else float(r[c])) for c in NUM},
             **{c: (None if nulo(r[c]) else str(r[c])) for c in CAT}} for r in df[NUM + CAT].to_dict("records")]


def invocar(runtime, endpoint, cuerpo, reintentos=6):
    """Una petición JSON → respuesta del contrato. Reintenta ante arranque en frío o throttling."""
    from botocore.exceptions import ClientError
    texto = json.dumps(cuerpo, allow_nan=False)
    for intento in range(reintentos):
        try:
            r = runtime.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Accept="application/json", Body=texto)
            return json.loads(r["Body"].read())
        except ClientError as e:
            codigo = e.response["Error"]["Code"]
            if codigo in ("ThrottlingException", "ModelError", "ServiceUnavailable", "InternalFailure") and intento < reintentos - 1:
                espera = 2 ** intento
                print(f"      {codigo}: reintento en {espera} s")
                time.sleep(espera)
                continue
            raise


def invocar_lote(runtime, endpoint, instancias):
    """Varias transacciones en una petición → vector de probabilidades (calibradas)."""
    resp = invocar(runtime, endpoint, {"instancias": instancias})
    p = np.array([r["probabilidad"] for r in resp["resultados"]], dtype=float)
    if len(p) != len(instancias):
        raise RuntimeError(f"El endpoint devolvió {len(p)} resultados para {len(instancias)} transacciones")
    return p


def predecir(invocador, instancias, lote, etiqueta):
    partes, t0 = [], time.perf_counter()
    for i in range(0, len(instancias), lote):
        partes.append(invocador(instancias[i:i + lote]))
        if sys.stdout.isatty():
            print(f"   {etiqueta}: {min(i + lote, len(instancias)):>6,}/{len(instancias):,} filas", end="\r", flush=True)
    seg = time.perf_counter() - t0
    print(f"   {etiqueta}: {len(instancias):,} filas en {len(partes)} peticiones · {seg:.1f} s ({1000 * seg / len(partes):.0f} ms/petición)")
    return np.concatenate(partes)


# ---------------------------------------------------------------- métricas del caso (misma lógica que entrenar.evaluar)
def calcular_metricas(ent, val, test, p_val, p_test):
    """val/test: dict con y, usd, hora, dura. p_* ya vienen calibrados por el endpoint.
    Los umbrales se buscan con VALIDACIÓN; las métricas se miden sobre PRUEBA."""
    wd_test = ent.pesos_dia(test["y"])
    umb, info = ent.buscar_umbrales(p_val, val["y"], val["usd"], val["hora"], val["dura"])
    m = ~test["dura"]
    r = ent.proyectar(test["y"][m], ent.decidir_zona(p_test, test["hora"], umb)[m], test["usd"][m], wd_test[m])
    met = {"roc_auc": float(roc_auc_score(test["y"], p_test)),
           "pr_auc_prod": float(average_precision_score(test["y"], p_test, sample_weight=wd_test)),
           "pr_auc_muestra": float(average_precision_score(test["y"], p_test)),
           "costo_anual_usd": r["costo_usd"], "fraude_no_detectado_usd": r["fraude_no_detectado_usd"],
           "fp_sobre_bloqueos": r["fp_sobre_bloqueos"], "bloqueos_dia": r["bloqueos_dia"],
           "revisiones_dia": r["revisiones_dia"], "deteccion": r["deteccion"],
           "n_fraudes_prueba": int(test["y"].sum()), "n_filas_prueba": int(len(test["y"]))}
    return met, umb, info


def conjunto(df, NUM, CAT):
    return {"inst": a_instancias(df, NUM, CAT), "y": df["y"].to_numpy(), "usd": df["monto_usd"].to_numpy(),
            "hora": df["hora_local"].to_numpy(), "dura": df["lista_negra_dura"].astype(bool).to_numpy()}


def mostrar(met, calidad, local):
    print("\n  métrica sobre PRUEBA (endpoint SageMaker)   valor     mínimo equipo   NAM local")
    for k in ("roc_auc", "pr_auc_prod"):
        minimo = calidad.get(f"{k}_min")
        ref = (local or {}).get("metricas_prueba", {}).get(k)
        print(f"  {k:<42} {met[k]:>7.3f} {('-' if minimo is None else f'{minimo}'):>14} {('-' if ref is None else f'{ref:.3f}'):>11}")
    print(f"  {'pr_auc_muestra':<42} {met['pr_auc_muestra']:>7.3f}")
    print(f"  {'costo anual proyectado (USD)':<42} {met['costo_anual_usd']:>10,.0f}")
    print(f"  {'fraude no detectado (USD/año)':<42} {met['fraude_no_detectado_usd']:>10,.0f}")
    print(f"  {'FP sobre bloqueos':<42} {met['fp_sobre_bloqueos']:>7.1%}   (objetivo ≤ 15%)")
    print(f"  {'bloqueos por día':<42} {met['bloqueos_dia']:>7.1f}")
    print(f"  {'revisiones por día':<42} {met['revisiones_dia']:>7.1f}   (capacidad 4.800)")
    print(f"  {'detección (fraude capturado)':<42} {met['deteccion']:>7.1%}")
    print(f"  ({met['n_fraudes_prueba']} fraudes en {met['n_filas_prueba']:,} filas: diferencias pequeñas no son diferencias)")


# ---------------------------------------------------------------- CloudWatch
def publicar_metricas(cw, endpoint, met):
    nombres = {"roc_auc": "ROC_AUC", "pr_auc_prod": "PR_AUC_Produccion", "costo_anual_usd": "CostoAnualUSD",
               "fraude_no_detectado_usd": "FraudeNoDetectadoUSD", "fp_sobre_bloqueos": "FP_sobre_Bloqueos",
               "bloqueos_dia": "BloqueosDia", "revisiones_dia": "RevisionesDia", "deteccion": "Deteccion"}
    datos = [{"MetricName": n, "Dimensions": [{"Name": "EndpointName", "Value": endpoint}], "Value": float(met[k]), "Unit": "None"}
             for k, n in nombres.items() if met.get(k) is not None and np.isfinite(met[k])]
    cw.put_metric_data(Namespace=NAMESPACE_PROPIO, MetricData=datos)
    return len(datos)


def leer_cloudwatch(cw, endpoint, desde, espera_s):
    """Métricas propias del endpoint (namespace AWS/SageMaker). Espera hasta que aparezcan las invocaciones."""
    dim = [{"Name": "EndpointName", "Value": endpoint}, {"Name": "VariantName", "Value": config.VARIANTE}]
    consultas = [("Invocations", "Sum", "Invocaciones"), ("ModelLatency", "Average", "Latencia del modelo prom. (µs)"),
                 ("ModelLatency", "Maximum", "Latencia del modelo máx. (µs)"), ("OverheadLatency", "Average", "Overhead prom. (µs)"),
                 ("ModelSetupTime", "Maximum", "Arranque en frío máx. (µs)"),
                 ("Invocation4XXErrors", "Sum", "Errores 4XX"), ("Invocation5XXErrors", "Sum", "Errores 5XX")]
    limite, resultado = time.time() + espera_s, {}
    while True:
        resultado = {}
        for metrica, estad, rotulo in consultas:
            pts = cw.get_metric_statistics(Namespace="AWS/SageMaker", MetricName=metrica, Dimensions=dim, StartTime=desde,
                                           EndTime=datetime.now(timezone.utc) + timedelta(minutes=1), Period=60, Statistics=[estad])["Datapoints"]
            if pts:
                vals = [p[estad] for p in pts]
                resultado[rotulo] = sum(vals) if estad == "Sum" else (max(vals) if estad == "Maximum" else sum(vals) / len(vals))
        if resultado.get("Invocaciones") or time.time() > limite:
            return resultado
        print("   CloudWatch aún no muestra las invocaciones (demora 1-3 min)...")
        time.sleep(20)


# ---------------------------------------------------------------- programa principal
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", help="por defecto el del paso 03")
    ap.add_argument("--lote", type=int, default=200, help="transacciones por petición (el límite del endpoint serverless es 4 MB)")
    ap.add_argument("--sin-cloudwatch", action="store_true", help="no publica ni lee métricas de CloudWatch")
    ap.add_argument("--espera-cw", type=int, default=240, help="segundos máximos esperando las métricas del endpoint")
    args = ap.parse_args(argv)

    est = config.leer_estado()
    endpoint = args.endpoint or config.requerir(est, "endpoint", "03_desplegar_endpoint.py")

    ent = config.importar_entrenar()
    d, NUM, CAT, FEATURES = config.cargar_tratado()
    _, va, te = config.separar(d)
    val, test = conjunto(va, NUM, CAT), conjunto(te, NUM, CAT)

    sess = config.sesion()
    runtime, cw = sess.client("sagemaker-runtime"), sess.client("cloudwatch")
    inicio = datetime.now(timezone.utc) - timedelta(minutes=1)
    invocador = lambda inst: invocar_lote(runtime, endpoint, inst)

    print(f"Endpoint: {endpoint} ({sess.region_name})")
    print("1/4 Validación → umbrales")
    p_val = predecir(invocador, val["inst"], args.lote, "validación")
    print("2/4 PRUEBA → métricas")
    p_test = predecir(invocador, test["inst"], args.lote, "prueba")

    met, umb, info = calcular_metricas(ent, val, test, p_val, p_test)
    calidad = json.loads((config.RAIZ / "servicio-ia" / "calidad.json").read_text())
    ruta_local = config.RAIZ / "servicio-ia" / "reporte_entrenamiento.json"
    local = json.loads(ruta_local.read_text()) if ruta_local.exists() else None
    mostrar(met, calidad, local)

    fallas = [f"{k} = {met[k]:.3f} < mínimo {calidad[k + '_min']}" for k in ("roc_auc", "pr_auc_prod") if not met[k] >= calidad[k + "_min"]]
    print("\n  puerta de calidad del equipo:", "APROBADA" if not fallas else "NO APROBADA → " + "; ".join(fallas))
    print("  umbrales (costo mínimo en validación):", json.dumps({k: round(v, 5) for k, v in umb.items()}))

    print("3/4 Invocación individual (contrato que consume servicio-ia)")
    i = int(np.argmax(test["y"] == 1))                               # primer fraude de la prueba
    t0 = time.perf_counter()
    ejemplo = invocar(runtime, endpoint, {"variables": test["inst"][i]})
    ms = 1000 * (time.perf_counter() - t0)
    top = dict(sorted(ejemplo["aportes"].items(), key=lambda kv: -abs(kv[1]))[:3])
    print(f"   {ms:.0f} ms · probabilidad {ejemplo['probabilidad']} · versión {ejemplo['version_modelo']} · "
          f"{len(ejemplo['aportes'])} aportes · razones (top 3): {top}")
    version = ejemplo["version_modelo"]

    salida = {"generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"), "endpoint": endpoint, "version_modelo": version,
              "metricas_prueba": met, "umbrales": umb, "umbrales_cumplen_restricciones_en_validacion": info,
              "aprobado": not fallas, "fallas": fallas, "ejemplo_invocacion": ejemplo, "latencia_ejemplo_ms": round(ms, 1),
              "hiperparametros": est.get("hiperparametros"), "memoria_mb": est.get("memoria_mb"), "concurrencia": est.get("concurrencia")}
    ruta = config.AQUI / "metricas_prueba_sagemaker.json"
    ruta.write_text(json.dumps(ent.a_python(salida), indent=2, ensure_ascii=False))
    umbrales = {"version": "u-sm-0001", "version_modelo": version, "umbrales": umb,
                "unidad": "probabilidad de fraude EN PRODUCCIÓN (ya calibrada por el endpoint)",
                "reglas_duras": ["lista_negra:solicitud_cumplimiento"], "creado_por": "aws/04_invocar_y_metricas.py (propuesta)",
                "creado_en": salida["generado_en"], "motivo": "mínimo costo en validación con revisiones ≤ capacidad de analistas y FP ≤ 15%",
                "autorizado_por": "PENDIENTE: gerencia de riesgo (DA5 / RO5)"}
    (config.AQUI / "umbrales_sagemaker.json").write_text(json.dumps(ent.a_python(umbrales), indent=2, ensure_ascii=False))
    print(f"  guardados: {ruta.name} y umbrales_sagemaker.json (en {config.AQUI})")

    if not args.sin_cloudwatch:
        print("4/4 CloudWatch")
        try:
            n = publicar_metricas(cw, endpoint, met)
            print(f"   {n} métricas del caso publicadas en el namespace {NAMESPACE_PROPIO} (CloudWatch → Metrics → All metrics)")
        except Exception as e:
            print(f"   (no pude publicar métricas propias: {type(e).__name__}: {e})")
        try:
            res = leer_cloudwatch(cw, endpoint, inicio, args.espera_cw)
            print("   métricas del endpoint (AWS/SageMaker):")
            for k, v in res.items():
                print(f"     {k:<34} {v:,.1f}")
            if not res:
                print("     (sin datos todavía: míralas en la consola en unos minutos)")
        except Exception as e:
            print(f"   (no pude leer CloudWatch: {type(e).__name__}: {e})")
    return 0 if not fallas else 1


if __name__ == "__main__":
    sys.exit(main())
