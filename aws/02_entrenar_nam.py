#!/usr/bin/env python3
"""
02_entrenar_nam.py · Lanza el entrenamiento de la NAM en SageMaker.   →  consola: SageMaker → Training jobs

Reemplaza al XGBoost integrado: la NAM no es un algoritmo integrado de SageMaker, así que se entrena con el
contenedor oficial de TensorFlow en "script mode" (aws/src/train_nam.py). El objetivo del caso 3 es el mismo
que el de XGBoost binary:logistic: salida sigmoide + entropía cruzada binaria (casos 1 y 3). Un caso multiclase
(multi:softprob + num_class) necesitaría una NAM de k salidas por factor: esta versión no la implementa.

Qué hace:
  1. Empaqueta aws/src/train_nam.py e inference.py en un sourcedir.tar.gz y lo sube al bucket.
  2. Crea el training job (canales train, validation y config en S3) con los hiperparámetros de abajo.
  3. Espera a que termine, muestra las métricas y guarda la ruta del model.tar.gz en aws/estado.json.

Uso:  python aws/02_entrenar_nam.py [--version 3.0.0] [--epocas 100] [--instancia ml.m5.xlarge] [--rol ARN] [--imagen URI] [--no-esperar]
"""
import argparse
import json
import sys
import tarfile
import time

import config


def hiperparametros(args):
    hp = {
        "objective": config.OBJETIVO,            # binary:logistic (casos 1 y 3)
        "version": args.version,                 # version_modelo que devuelve el endpoint
        "epochs": args.epocas,
        "learning_rate": args.lr,
        "batch_size": args.batch,
        "l2": args.l2,
        "capas": args.capas,                     # neuronas de las capas ocultas de cada subred
        "peso_fraude": args.peso_fraude,
        "peso_no_revisadas": 1.0,
        "semilla": 42,
        "eval_cada": 10,
    }
    if config.OBJETIVO != "binary:logistic":
        sys.exit("ERROR: esta NAM solo implementa binary:logistic (casos 1 y 3).")
    return hp


def empaquetar_codigo(s3, bucket, prefijo, nombre_trabajo):
    ruta = config.LOCAL / "sourcedir.tar.gz"
    config.LOCAL.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ruta, "w:gz") as t:
        t.add(config.AQUI / "src" / "train_nam.py", arcname="train_nam.py")
        t.add(config.AQUI / "src" / "inference.py", arcname="inference.py")      # train_nam.py lo copia a code/ del modelo
        t.add(config.AQUI / "src" / "requirements.txt", arcname="requirements.txt")
    key = f"{prefijo}/codigo/{nombre_trabajo}/sourcedir.tar.gz"
    s3.upload_file(str(ruta), bucket, key)
    return f"s3://{bucket}/{key}"


def esperar(sm, nombre, cada=30):
    t0 = time.time()
    while True:
        d = sm.describe_training_job(TrainingJobName=nombre)
        estado, sub = d["TrainingJobStatus"], d.get("SecondaryStatus", "")
        print(f"   [{int(time.time() - t0):>4} s] {estado} · {sub}")
        if estado in ("Completed", "Failed", "Stopped"):
            return d
        time.sleep(cada)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="3.0.0", help="version_modelo del endpoint (aparece en cada respuesta)")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--capas", default="16,8")
    ap.add_argument("--peso-fraude", type=float, default=20.0)
    ap.add_argument("--instancia", default="ml.m5.xlarge", help="tipo de instancia de entrenamiento (CPU)")
    ap.add_argument("--rol", help="ARN del rol de SageMaker (por defecto LabRole)")
    ap.add_argument("--imagen", help="URI del contenedor de entrenamiento (por defecto TensorFlow 2.16 oficial)")
    ap.add_argument("--no-esperar", action="store_true")
    args = ap.parse_args(argv)

    est = config.leer_estado()
    bucket = config.requerir(est, "bucket", "01_s3_subir_datos.py")
    prefijo = est.get("prefijo_s3", config.PREFIJO_S3)
    sess = config.sesion()
    region = sess.region_name
    sm, s3 = sess.client("sagemaker"), sess.client("s3")
    rol = config.rol_ejecucion(sess, args.rol)
    imagen = config.uri_imagen(region, "training", args.imagen)

    nombre = f"ity1102-ep3-nam-{config.marca_tiempo()}"
    hp = hiperparametros(args)
    print(f"Training job: {nombre}\n   imagen: {imagen}\n   rol: {rol}\n   instancia: {args.instancia}")
    print("   hiperparámetros:", json.dumps(hp, ensure_ascii=False))

    codigo = empaquetar_codigo(s3, bucket, prefijo, nombre)
    # El contenedor de framework espera cada hiperparámetro como JSON, más estas 4 claves de "script mode"
    hp_sm = {k: json.dumps(v) for k, v in hp.items()}
    hp_sm.update({"sagemaker_program": json.dumps("train_nam.py"), "sagemaker_submit_directory": json.dumps(codigo),
                  "sagemaker_region": json.dumps(region), "sagemaker_container_log_level": "20"})

    def canal(nombre_canal, subprefijo, tipo):
        return {"ChannelName": nombre_canal, "ContentType": tipo, "InputMode": "File",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3DataDistributionType": "FullyReplicated",
                                                "S3Uri": f"s3://{bucket}/{prefijo}/{subprefijo}/"}}}

    sm.create_training_job(
        TrainingJobName=nombre,
        RoleArn=rol,
        AlgorithmSpecification={
            "TrainingImage": imagen, "TrainingInputMode": "File",
            "MetricDefinitions": [{"Name": "val:pr_auc_prod", "Regex": r"val_pr_auc_prod=([0-9\.]+)"}]},   # → CloudWatch
        InputDataConfig=[canal("train", "train", "text/csv"), canal("validation", "validation", "text/csv"),
                         canal("config", "config", "application/json")],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/{prefijo}/salida/"},
        ResourceConfig={"InstanceType": args.instancia, "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters=hp_sm,
        Tags=[{"Key": "proyecto", "Value": "ity1102-ep3"}, {"Key": "modelo", "Value": "nam"}])
    config.guardar_estado(trabajo_entrenamiento=nombre, hiperparametros=hp, imagen_entrenamiento=imagen, rol_arn=rol)
    print("Lanzado. Míralo en la consola: SageMaker → Training jobs →", nombre)
    if args.no_esperar:
        return 0

    d = esperar(sm, nombre)
    if d["TrainingJobStatus"] != "Completed":
        sys.exit(f"ERROR: el entrenamiento terminó en {d['TrainingJobStatus']}: {d.get('FailureReason', '(sin detalle)')}\n"
                 f"Detalle en CloudWatch → Logs → /aws/sagemaker/TrainingJobs → {nombre}")
    artefacto = d["ModelArtifacts"]["S3ModelArtifacts"]
    metricas = {m["MetricName"]: m["Value"] for m in d.get("FinalMetricDataList", [])}
    config.guardar_estado(model_artifact=artefacto, metricas_entrenamiento=metricas,
                          segundos_entrenamiento=d.get("TrainingTimeInSeconds"))
    print(f"\nEntrenamiento completo en {d.get('TrainingTimeInSeconds')} s · métricas: {metricas}")
    print(f"Modelo: {artefacto}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
