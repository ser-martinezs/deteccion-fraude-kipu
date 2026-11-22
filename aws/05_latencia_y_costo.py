#!/usr/bin/env python3
"""
05_latencia_y_costo.py · Mide el tiempo de invocación del endpoint, el arranque en frío y el costo mensual estimado.
                         →  guía sección 7 · consola: CloudWatch → Metrics → SageMaker → Endpoint

Qué mide (la latencia incluye la red del lugar desde donde lo ejecutes: CloudShell, tu PC o la EC2; para la
medición del informe, ejecútalo desde la EC2 del sistema, que es de donde llama servicio-ia):
  1. Arranque en frío: espera --frio-min minutos SIN usar el endpoint (la guía pide 10 o más), invoca y de inmediato
     invoca otra vez. Anota los dos tiempos.
  2. Latencia en caliente: --n invocaciones de UNA transacción (--hilos en paralelo): p50, p90, p95, p99, y si el p99
     cumple el objetivo (--p99-ms, 100 por defecto).
  3. CloudWatch, lado servidor: ModelLatency, OverheadLatency, arranques en frío (ModelSetupTime) y errores.
  4. Costo mensual de SageMaker serverless al volumen del caso (200.000 transacciones/día) y la concurrencia que
     exige el pico (5.500/min), con la opción de concurrencia aprovisionada.
  5. --comparar-memoria 2048,3072: actualiza el MISMO endpoint a cada tamaño (queda en frío: se mide el arranque),
     mide y propone el más barato que cumpla el p99. Al terminar el endpoint queda en el último tamaño probado
     (usa --aplicar-mejor para dejarlo en el recomendado).

Uso:
  python aws/05_latencia_y_costo.py --rapido                        # 20 invocaciones, sin espera ni CloudWatch (para la defensa)
  python aws/05_latencia_y_costo.py                                 # completo: espera 12 min sin uso y mide
  python aws/05_latencia_y_costo.py --comparar-memoria 2048,3072 --aplicar-mejor
Guarda aws/latencia_<fecha>.json con todo lo medido.
"""
import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

import config

# ---- precios de SageMaker Serverless Inference (USD). Fuente: aws.amazon.com/sagemaker/ai/pricing, ejemplos #13 y #14
#      (página actualizada el 17-nov-2026): 2 GB = 0,00004 USD/s de cómputo; datos 0,016 USD/GB entrada+salida;
#      concurrencia aprovisionada 0,00001 USD/s y duración atendida con ella 0,000023 USD/s (ambos para 2 GB).
#      Para los demás tamaños se asume escala LINEAL por GB: la página no publica la tabla completa. Confírmalo en la
#      Calculadora de AWS antes de poner la cifra en el informe. No incluye la capa gratuita (150.000 s los 2 primeros meses).
PRECIO_GB_S = 0.00002
PRECIO_GB_DATOS = 0.016
PC_GB_S = 0.000005
PC_DURACION_GB_S = 0.0000115
PICO_RPS = 5500 / 60            # 5.500 transacciones por minuto en el peak del caso (comentario de servicio-gateway/nginx.conf)
DIAS_MES = 30


def costo_mensual(solicitudes, ms_por_solicitud, memoria_mb, bytes_por_solicitud, pc=0, horas_pc=730):
    """USD al mes. pc = concurrencia aprovisionada (0 = solo bajo demanda); horas_pc = horas del mes que se mantiene."""
    gb = memoria_mb / 1024
    seg = solicitudes * ms_por_solicitud / 1000
    datos = solicitudes * bytes_por_solicitud / 1e9 * PRECIO_GB_DATOS
    if pc:
        computo, aprovisionado = seg * gb * PC_DURACION_GB_S, pc * horas_pc * 3600 * gb * PC_GB_S
    else:
        computo, aprovisionado = seg * gb * PRECIO_GB_S, 0.0
    return {"computo": computo, "aprovisionado": aprovisionado, "datos": datos, "total": computo + aprovisionado + datos}


# ---------------------------------------------------------------- medición
def cliente_medicion(sess):
    """Cliente SIN reintentos y con timeout largo: así el arranque en frío se ve completo y no queda escondido."""
    from botocore.config import Config
    return sess.client("sagemaker-runtime", config=Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 0}))


def invocar_una(rt, endpoint, cuerpo):
    t0 = time.perf_counter()
    try:
        r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Accept="application/json", Body=cuerpo)
        salida = r["Body"].read()
        return {"ms": (time.perf_counter() - t0) * 1000, "ok": True, "bytes": len(cuerpo) + len(salida)}
    except Exception as e:
        respuesta = getattr(e, "response", None)
        codigo = respuesta["Error"]["Code"] if isinstance(respuesta, dict) and "Error" in respuesta else type(e).__name__
        return {"ms": (time.perf_counter() - t0) * 1000, "ok": False, "error": codigo}


def serie_caliente(rt, endpoint, cuerpos, n, hilos):
    tareas = [cuerpos[i % len(cuerpos)] for i in range(n)]
    t0 = time.perf_counter()
    if hilos <= 1:
        res = [invocar_una(rt, endpoint, c) for c in tareas]
    else:
        with ThreadPoolExecutor(hilos) as ex:
            res = list(ex.map(lambda c: invocar_una(rt, endpoint, c), tareas))
    return res, time.perf_counter() - t0


def estadisticas(res, segundos, p99_objetivo):
    oks = np.array([r["ms"] for r in res if r["ok"]])
    errores = dict(Counter(r["error"] for r in res if not r["ok"]))
    if len(oks) == 0:
        return {"n": len(res), "ok": 0, "errores": errores}
    p50, p90, p95, p99 = np.percentile(oks, [50, 90, 95, 99])
    return {"n": len(res), "ok": int(len(oks)), "errores": errores, "media_ms": float(oks.mean()), "min_ms": float(oks.min()),
            "p50_ms": float(p50), "p90_ms": float(p90), "p95_ms": float(p95), "p99_ms": float(p99), "max_ms": float(oks.max()),
            "solicitudes_por_s": len(res) / segundos if segundos > 0 else None, "cumple_p99": bool(p99 <= p99_objetivo),
            "bytes_por_solicitud": float(np.mean([r["bytes"] for r in res if r["ok"]]))}


def esperar_minutos(minutos):
    fin, tty = time.time() + minutos * 60, sys.stdout.isatty()
    ultimo = 0
    while time.time() < fin:
        resto = fin - time.time()
        if tty or time.time() - ultimo >= 60:
            print(f"   sin usar el endpoint... faltan {int(resto // 60)} min {int(resto % 60):02d} s", end="\r" if tty else "\n", flush=True)
            ultimo = time.time()
        time.sleep(min(15, max(resto, 0)))
    if tty:
        print(" " * 60, end="\r")


# ---------------------------------------------------------------- CloudWatch (lado servidor)
def _leer_cw(cw, endpoint, desde):
    dim = [{"Name": "EndpointName", "Value": endpoint}, {"Name": "VariantName", "Value": config.VARIANTE}]
    hasta = datetime.now(timezone.utc) + timedelta(minutes=1)
    periodo = 60 * max(1, math.ceil((hasta - desde).total_seconds() / 60))

    def pedir(metrica, estadisticas=None, extendidas=None):
        kw = dict(Namespace="AWS/SageMaker", MetricName=metrica, Dimensions=dim, StartTime=desde, EndTime=hasta, Period=periodo)
        if estadisticas:
            kw["Statistics"] = estadisticas
        if extendidas:
            kw["ExtendedStatistics"] = extendidas               # los percentiles se piden aparte de las demás estadísticas
        return cw.get_metric_statistics(**kw)["Datapoints"]

    out = {"invocaciones": sum(p["Sum"] for p in pedir("Invocations", ["Sum"]))}
    for metrica, clave in (("ModelLatency", "modelo"), ("OverheadLatency", "overhead")):
        pts = pedir(metrica, ["Average", "Maximum", "SampleCount"])
        if pts:
            n = sum(p["SampleCount"] for p in pts) or 1
            out[f"{clave}_prom_ms"] = sum(p["Average"] * p["SampleCount"] for p in pts) / n / 1000     # CloudWatch las entrega en µs
            out[f"{clave}_max_ms"] = max(p["Maximum"] for p in pts) / 1000
        pct = pedir(metrica, extendidas=["p50", "p99"])
        if pct:
            out[f"{clave}_p50_ms"] = max(p["ExtendedStatistics"]["p50"] for p in pct) / 1000
            out[f"{clave}_p99_ms"] = max(p["ExtendedStatistics"]["p99"] for p in pct) / 1000
    pts = pedir("ModelSetupTime", ["Maximum", "SampleCount"])
    if pts:
        out["arranques_en_frio"] = int(sum(p["SampleCount"] for p in pts))
        out["arranque_frio_max_ms"] = max(p["Maximum"] for p in pts) / 1000
    for metrica, clave in (("Invocation4XXErrors", "errores_4xx"), ("Invocation5XXErrors", "errores_5xx")):
        out[clave] = sum(p["Sum"] for p in pedir(metrica, ["Sum"]))
    return out


def cloudwatch_servidor(cw, endpoint, desde, esperadas, espera_s):
    limite = time.time() + espera_s
    while True:
        res = _leer_cw(cw, endpoint, desde)
        if res["invocaciones"] >= esperadas or time.time() > limite:
            return res
        print(f"   CloudWatch muestra {res['invocaciones']:.0f} de {esperadas} invocaciones (demora 1-3 min)...")
        time.sleep(20)


# ---------------------------------------------------------------- una medición completa
def medir(sess, endpoint, cuerpos, args, memoria_mb, concurrencia, solicitudes_mes, frio_min):
    rt, cw = cliente_medicion(sess), sess.client("cloudwatch")
    desde = datetime.now(timezone.utc) - timedelta(minutes=1)
    if frio_min > 0:
        print(f"   Arranque en frío: espero {frio_min:g} min sin invocar (el endpoint serverless se enfría)...")
        esperar_minutos(frio_min)
    a = invocar_una(rt, endpoint, cuerpos[0])
    b = invocar_una(rt, endpoint, cuerpos[0])
    frio = {"espera_min": frio_min, "primera_ms": a["ms"], "primera_ok": a["ok"], "primera_error": a.get("error"),
            "segunda_ms": b["ms"], "segunda_ok": b["ok"], "segunda_error": b.get("error")}
    txt = lambda r: f"{r['ms']:,.0f} ms" if r["ok"] else f"ERROR {r['error']} ({r['ms']:,.0f} ms)"
    print(f"   Primera invocación: {txt(a)} · segunda, de inmediato: {txt(b)}"
          + (f"   [tras {frio_min:g} min sin uso]" if frio_min > 0 else "   [sin espera previa: la primera es fría solo si el endpoint estaba ocioso]"))

    res, seg = serie_caliente(rt, endpoint, cuerpos, args.n, args.hilos)
    est = estadisticas(res, seg, args.p99_ms)
    if est["ok"]:
        print(f"   En caliente (n={est['n']}, {args.hilos} hilo(s)): p50 {est['p50_ms']:.0f} · p90 {est['p90_ms']:.0f} · p95 {est['p95_ms']:.0f} · "
              f"p99 {est['p99_ms']:.0f} · máx {est['max_ms']:.0f} ms · errores {sum(est['errores'].values())}"
              f"  →  p99 ≤ {args.p99_ms:g} ms: {'CUMPLE' if est['cumple_p99'] else 'NO CUMPLE'}")
        if est["n"] < 500:
            print(f"   (con n={est['n']} el p99 se apoya en {max(1, round(est['n'] * 0.01))} observación(es): para el informe usa --n 1000)")
    else:
        print(f"   Todas las invocaciones fallaron: {est['errores']}")
        return {"memoria_mb": memoria_mb, "frio": frio, "caliente": est}

    servidor = None
    if not args.sin_cloudwatch:
        try:
            servidor = cloudwatch_servidor(cw, endpoint, desde, esperadas=2 + args.n, espera_s=args.espera_cw)
            partes = [f"invocaciones {servidor['invocaciones']:.0f}"]
            for k, rot in (("modelo_prom_ms", "ModelLatency prom"), ("modelo_p99_ms", "p99"), ("overhead_prom_ms", "OverheadLatency prom")):
                if k in servidor:
                    partes.append(f"{rot} {servidor[k]:.1f} ms")
            if "arranques_en_frio" in servidor:
                partes.append(f"arranques en frío {servidor['arranques_en_frio']} (máx {servidor['arranque_frio_max_ms']:,.0f} ms)")
            partes.append(f"errores 4XX/5XX {servidor['errores_4xx']:.0f}/{servidor['errores_5xx']:.0f}")
            print("   CloudWatch (servidor): " + " · ".join(partes))
        except Exception as e:
            print(f"   (no pude leer CloudWatch: {type(e).__name__}: {e})")

    # costo y concurrencia
    if servidor and "modelo_prom_ms" in servidor:
        ms_fact, fuente = servidor["modelo_prom_ms"] + servidor.get("overhead_prom_ms", 0.0), "ModelLatency+OverheadLatency de CloudWatch"
    else:
        ms_fact, fuente = est["p50_ms"], "p50 medido desde el cliente (incluye la red: sobreestima)"
    conc_necesaria = max(1, math.ceil(PICO_RPS * ms_fact / 1000))                         # ley de Little: llegadas/s × tiempo de servicio
    costo = costo_mensual(solicitudes_mes, ms_fact, memoria_mb, est["bytes_por_solicitud"])
    costo_pc = costo_mensual(solicitudes_mes, ms_fact, memoria_mb, est["bytes_por_solicitud"], pc=conc_necesaria)
    print(f"   Costo mensual estimado ({solicitudes_mes:,.0f} solicitudes, {memoria_mb} MB, ~{ms_fact:.0f} ms facturados por {fuente}): "
          f"USD {costo['total']:,.2f}  (cómputo {costo['computo']:,.2f} + datos {costo['datos']:,.2f})")
    print(f"   Con concurrencia aprovisionada de {conc_necesaria} las 24 h: USD {costo_pc['total']:,.2f}  (aprovisionado {costo_pc['aprovisionado']:,.2f})")
    ok_conc = concurrencia >= conc_necesaria
    print(f"   Concurrencia en el pico ({PICO_RPS:.0f} req/s × {ms_fact:.0f} ms) ≈ {conc_necesaria}; el endpoint admite {concurrencia}: "
          f"{'alcanza' if ok_conc else 'NO ALCANZA → habrá throttling en el pico; sube --concurrencia en el paso 03'}")
    return {"memoria_mb": memoria_mb, "concurrencia_configurada": concurrencia, "frio": frio, "caliente": est, "servidor": servidor,
            "ms_facturados": ms_fact, "fuente_ms": fuente, "concurrencia_necesaria_pico": conc_necesaria,
            "costo_mensual_usd": costo, "costo_mensual_con_pc_usd": costo_pc, "solicitudes_mes": solicitudes_mes}


# ---------------------------------------------------------------- programa principal
def preparar_cuerpos(n):
    """Transacciones reales de la PRUEBA (las de la medición no se usan para decidir nada)."""
    s04 = config.importar_script("04_invocar_y_metricas")
    d, NUM, CAT, _ = config.cargar_tratado()
    _, _, _, te = config.cargar_corte(d)
    muestra = te.sample(min(len(te), max(n, 50)), random_state=0)
    return [json.dumps({"variables": i}, allow_nan=False).encode() for i in s04.a_instancias(muestra, NUM, CAT)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", help="por defecto el del paso 03")
    ap.add_argument("--rapido", action="store_true", help="20 invocaciones, sin espera en frío ni CloudWatch")
    ap.add_argument("--frio-min", type=float, default=12, help="minutos sin usar el endpoint antes de medir el arranque en frío (0 = sin espera)")
    ap.add_argument("--n", type=int, default=300, help="invocaciones en caliente (1000 para el informe)")
    ap.add_argument("--hilos", type=int, default=1, help="invocaciones en paralelo (más de 1 prueba la concurrencia)")
    ap.add_argument("--p99-ms", type=float, default=100, help="objetivo de p99")
    ap.add_argument("--comparar-memoria", help="tamaños separados por coma, p. ej. 2048,3072")
    ap.add_argument("--concurrencia", type=int, help="MaxConcurrency al reconfigurar (por defecto la del paso 03)")
    ap.add_argument("--aplicar-mejor", action="store_true", help="con --comparar-memoria, deja el endpoint en el tamaño recomendado")
    ap.add_argument("--solicitudes-mes", type=float, help="por defecto 200.000 al día × 30")
    ap.add_argument("--sin-cloudwatch", action="store_true")
    ap.add_argument("--espera-cw", type=int, default=240, help="segundos máximos esperando a CloudWatch")
    args = ap.parse_args(argv)
    if args.rapido:
        args.frio_min, args.n, args.sin_cloudwatch = 0, 20, True

    est = config.leer_estado()
    endpoint = args.endpoint or config.requerir(est, "endpoint", "03_desplegar_endpoint.py")
    memoria = int(est.get("memoria_mb", 2048))
    concurrencia = args.concurrencia or int(est.get("concurrencia", 5))
    solicitudes_mes = args.solicitudes_mes or config.importar_entrenar().TX_DIA * DIAS_MES
    sess = config.sesion()
    cuerpos = preparar_cuerpos(args.n)
    print(f"Endpoint {endpoint} ({sess.region_name}) · {len(cuerpos)} transacciones de la prueba · objetivo p99 ≤ {args.p99_ms:g} ms")

    mediciones = []
    if not args.comparar_memoria:
        print(f"\n== {memoria} MB · concurrencia {concurrencia} ==")
        mediciones.append(medir(sess, endpoint, cuerpos, args, memoria, concurrencia, solicitudes_mes, args.frio_min))
    else:
        tamanos = [int(x) for x in args.comparar_memoria.split(",")]
        if any(t not in (1024, 2048, 3072, 4096, 5120, 6144) for t in tamanos):
            sys.exit("ERROR: los tamaños válidos son 1024, 2048, 3072, 4096, 5120 y 6144 MB")
        desplegar = config.importar_script("03_desplegar_endpoint")
        for t in tamanos:
            print(f"\n== Reconfigurando {endpoint} a {t} MB (queda en frío: se mide el arranque) ==")
            desplegar.main(["--endpoint", endpoint, "--memoria", str(t), "--concurrencia", str(concurrencia), "--limpiar"])
            mediciones.append(medir(sess, endpoint, cuerpos, args, t, concurrencia, solicitudes_mes, 0))

    if args.comparar_memoria:
        print("\n== Comparación de memoria ==")
        print(f"  {'MB':>5} {'1.ª invoc. (ms)':>16} {'p50':>6} {'p99':>6} {'p99 ≤ obj.':>11} {'USD/mes':>10} {'conc. pico':>11}")
        for m in mediciones:
            c = m["caliente"]
            if not c.get("ok"):
                print(f"  {m['memoria_mb']:>5}  (sin mediciones válidas)")
                continue
            print(f"  {m['memoria_mb']:>5} {m['frio']['primera_ms']:>16,.0f} {c['p50_ms']:>6.0f} {c['p99_ms']:>6.0f} {'sí' if c['cumple_p99'] else 'NO':>11} "
                  f"{m['costo_mensual_usd']['total']:>10,.2f} {m['concurrencia_necesaria_pico']:>11}")
        cumplen = [m for m in mediciones if m["caliente"].get("cumple_p99")]
        mejor = min(cumplen, key=lambda m: m["costo_mensual_usd"]["total"]) if cumplen else None
        if mejor:
            print(f"\n  Recomendado: {mejor['memoria_mb']} MB (el más barato que cumple p99 ≤ {args.p99_ms:g} ms).")
            if args.aplicar_mejor and mejor["memoria_mb"] != mediciones[-1]["memoria_mb"]:
                print(f"  Aplicando {mejor['memoria_mb']} MB al endpoint...")
                desplegar.main(["--endpoint", endpoint, "--memoria", str(mejor["memoria_mb"]), "--concurrencia", str(concurrencia), "--limpiar"])
            elif mejor["memoria_mb"] != mediciones[-1]["memoria_mb"]:
                print(f"  El endpoint quedó en {mediciones[-1]['memoria_mb']} MB: aplica el recomendado con  python aws/03_desplegar_endpoint.py --memoria {mejor['memoria_mb']}")
        else:
            print(f"\n  Ningún tamaño cumple p99 ≤ {args.p99_ms:g} ms en serverless bajo demanda. Opciones, cada una con su costo (ver arriba):")
            print("   · concurrencia aprovisionada (mantiene el modelo caliente, elimina el arranque en frío);")
            print("   · endpoint de tiempo real (ml.m5.large / ml.t2.medium, cobra por hora encendido);")
            print("   · volver el modelo al contenedor de servicio-ia (sin llamada de red).")
        if len(cumplen) == len(mediciones) and len(mediciones) > 1:
            print("  Todos cumplen: la memoria extra solo agrega costo si no baja el p99 de forma apreciable.")

    ruta = config.AQUI / f"latencia_{config.marca_tiempo()}.json"
    ruta.write_text(json.dumps({"generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"), "endpoint": endpoint,
                                "region": sess.region_name, "p99_objetivo_ms": args.p99_ms, "hilos": args.hilos,
                                "pico_rps": PICO_RPS, "mediciones": mediciones}, indent=2, ensure_ascii=False, default=float))
    print(f"\nGuardado: {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
