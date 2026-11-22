#!/usr/bin/env python3
"""
01_s3_subir_datos.py · Crea el bucket y sube los CSV (la PRUEBA NO SE SUBE).   →  consola: S3 → ity1102-ep3-…

Qué hace:
  1. Lee datos/tratado/variables_modelo.csv y separa train / validación / prueba ANTES de entrenar:
       --corte estratificado (por defecto): prueba = 20 % apartado al azar y estratificado por y (random_state=42),
                                            el resto 75/25 en train y validación  → 60/20/20 con igual tasa de fraude.
       --corte fecha: el corte temporal de EP2 (columna split de preparar.py).
     El corte queda en aws/local/corte.json: el paso 04 evalúa exactamente sobre esas filas.
  2. Ajusta el preprocesador SOLO con entrenamiento (el mismo de entrenar.py) y lo guarda en aws/local/.
  3. Genera train.csv y validation.csv ya transformados: columnas  y, etiquetada, x000, x001, ...
     (la prueba se transforma después, en local, en el script 04: nunca sale de tu computador).
  4. Crea el bucket  ity1102-ep3-<id de cuenta>  si no existe (idempotente) y sube:
        s3://<bucket>/fraude-nam/train/train.csv
        s3://<bucket>/fraude-nam/validation/validation.csv
        s3://<bucket>/fraude-nam/config/grupos.json         (qué columnas forman cada factor de la NAM)
        s3://<bucket>/fraude-nam/config/preprocesador.json  (el preprocesador en JSON: lo aplica inference.py en el endpoint)
        s3://<bucket>/fraude-nam/artefactos/preprocesador.joblib
  5. Verifica que en el bucket no haya ningún archivo de prueba.

Uso:  python aws/01_s3_subir_datos.py [--corte estratificado|fecha] [--bucket NOMBRE] [--sufijo TEXTO]
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

import config


def exportar_preprocesador(prep, NUM, CAT):
    """ColumnTransformer de entrenar.py → dict JSON con lo mínimo para reproducirlo en numpy (src/inference.py)."""
    pn, pc = prep.named_transformers_["num"], prep.named_transformers_["cat"]
    imp, esc = pn.named_steps["imputar"], pn.named_steps["escalar"]
    onehot, imp_c = pc.named_steps["onehot"], pc.named_steps["imputar"]
    if len(imp.statistics_) != len(NUM) or np.isnan(imp.statistics_).any():
        sys.exit("ERROR: el imputador descartó o dejó sin mediana alguna variable numérica; revisa datos/tratado.")
    indicadores = [int(i) for i in imp.indicator_.features_] if imp.indicator_ is not None else []
    return {"numericas": list(NUM), "categoricas": list(CAT),
            "medianas": [float(v) for v in imp.statistics_], "indicadores": indicadores,
            "media": [float(v) for v in esc.mean_], "escala": [float(v) for v in esc.scale_],
            "categorias": {c: [str(x) for x in cats] for c, cats in zip(CAT, onehot.categories_)},
            "relleno_cat": str(imp_c.fill_value)}


def generar_csv(tratado, destino, corte="estratificado"):
    """Escribe train.csv / validation.csv transformados y los archivos de apoyo. Devuelve un resumen."""
    ent = config.importar_entrenar()
    d, NUM, CAT, FEATURES = config.cargar_tratado(tratado)
    tr, va, te = config.separar_corte(d, corte)
    config.guardar_corte(corte, tr, va, te)

    prep = ent.construir_preprocesador(NUM, CAT).fit(tr[FEATURES])           # SOLO con entrenamiento
    nombres = list(prep.get_feature_names_out())
    grupos = ent.grupos_de(nombres, CAT)                                     # factor -> índices de columnas

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    columnas = [f"x{j:03d}" for j in range(len(nombres))]
    cabecera = ",".join(["y", "etiquetada"] + columnas)

    archivos = {}
    for nombre, df in (("train", tr), ("validation", va)):
        A = prep.transform(df[FEATURES]).astype(np.float32)
        M = np.column_stack([df["y"].to_numpy(dtype=np.float32), df["etiquetada"].to_numpy(dtype=np.float32), A])
        ruta = destino / f"{nombre}.csv"
        np.savetxt(ruta, M, delimiter=",", header=cabecera, comments="", fmt="%.7g")
        archivos[nombre] = ruta

    pj = exportar_preprocesador(prep, NUM, CAT)
    if len(NUM) + len(pj["indicadores"]) + sum(len(v) for v in pj["categorias"].values()) != len(nombres):
        sys.exit("ERROR: el preprocesador exportado no produce las mismas columnas que el original.")
    (destino / "preprocesador.json").write_text(json.dumps(pj, ensure_ascii=False))
    (destino / "grupos.json").write_text(json.dumps(grupos, ensure_ascii=False))
    (destino / "columnas.json").write_text(json.dumps(dict(zip(columnas, nombres)), indent=1, ensure_ascii=False))
    config.LOCAL.mkdir(parents=True, exist_ok=True)
    joblib.dump(prep, config.LOCAL / "preprocesador.joblib")
    archivos["grupos"] = destino / "grupos.json"
    archivos["preprocesador_json"] = destino / "preprocesador.json"
    archivos["preprocesador"] = config.LOCAL / "preprocesador.joblib"
    resumen = {"corte": corte, "n_entrada": len(nombres), "n_factores": len(grupos),
               "filas": {"train": len(tr), "validation": len(va), "test (NO se sube)": len(te)},
               "fraudes": {"train": int(tr["y"].sum()), "validation": int(va["y"].sum()), "test (NO se sube)": int(te["y"].sum())}}
    return archivos, resumen


def asegurar_bucket(s3, nombre, region):
    from botocore.exceptions import ClientError
    try:
        s3.head_bucket(Bucket=nombre)
        print(f"   el bucket {nombre} ya existe: lo reutilizo")
        return
    except ClientError as e:
        codigo = str(e.response["Error"]["Code"])
        if codigo == "403":
            sys.exit(f"ERROR: el nombre {nombre} ya pertenece a otra cuenta (los nombres de S3 son globales). Usa --sufijo.")
        if codigo not in ("404", "NoSuchBucket", "NotFound"):
            raise
    args = {"Bucket": nombre}
    if region != "us-east-1":
        args["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**args)
    print(f"   bucket {nombre} creado en {region}")
    try:
        s3.put_public_access_block(Bucket=nombre, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    except ClientError as e:
        print(f"   (no pude bloquear el acceso público: {e.response['Error']['Code']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tratado", type=Path, default=config.TRATADO)
    ap.add_argument("--corte", choices=("estratificado", "fecha"), default="estratificado",
                    help="cómo se aparta la prueba (por defecto estratificado, como pide la guía)")
    ap.add_argument("--bucket", help=f"nombre completo (por defecto {config.PREFIJO_BUCKET}-<id de cuenta>)")
    ap.add_argument("--sufijo", help="reemplaza al id de cuenta en el nombre del bucket")
    args = ap.parse_args(argv)

    print("1/3 Preparando CSV locales (preprocesador ajustado solo con entrenamiento)...")
    archivos, resumen = generar_csv(args.tratado, config.LOCAL / "s3", args.corte)
    print("   ", json.dumps(resumen, ensure_ascii=False))

    sess = config.sesion()
    region = sess.region_name
    bucket = args.bucket or f"{config.PREFIJO_BUCKET}-{args.sufijo or config.id_cuenta(sess)}".lower()
    s3 = sess.client("s3")

    print(f"2/3 Bucket ({region})...")
    asegurar_bucket(s3, bucket, region)

    print("3/3 Subiendo (la prueba NO se sube)...")
    p = config.PREFIJO_S3
    destinos = {"train": f"{p}/train/train.csv", "validation": f"{p}/validation/validation.csv",
                "grupos": f"{p}/config/grupos.json", "preprocesador_json": f"{p}/config/preprocesador.json",
                "preprocesador": f"{p}/artefactos/preprocesador.joblib"}
    for clave, key in destinos.items():
        s3.upload_file(str(archivos[clave]), bucket, key)
        print(f"   s3://{bucket}/{key}")

    claves = [o["Key"] for pag in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=p)
              for o in pag.get("Contents", [])]
    if any("test" in k.lower() or "prueba" in k.lower() for k in claves):
        sys.exit(f"ERROR: hay archivos de prueba en el bucket: {[k for k in claves if 'test' in k.lower() or 'prueba' in k.lower()]}")

    config.guardar_estado(region=region, bucket=bucket, prefijo_s3=p, n_entrada=resumen["n_entrada"], corte=args.corte)
    print(f"\nListo. Revisa en la consola: S3 → {bucket} → {p}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
