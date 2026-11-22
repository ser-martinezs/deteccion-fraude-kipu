"""
aws/config.py · Configuración y utilidades comunes de los 4 scripts (EP3 · caso 3 · fraude · NAM).

Los scripts se encadenan a través de aws/estado.json (bucket, trabajo de entrenamiento, artefacto del modelo,
endpoint...), así que se ejecutan en orden: 01 → 02 → 03 → 04. Cada uno es idempotente.

Esta carpeta (aws/) va dentro del repositorio, junto a datos/ y servicio-ia/: reutiliza las funciones de
servicio-ia/entrenar.py (preprocesador, factores, calibración, umbrales, costo) para que las métricas sean
las mismas que las del entrenamiento local.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
RUTA_ESTADO = AQUI / "estado.json"
LOCAL = AQUI / "local"                      # preprocesador y archivos que NO se suben a S3
RUTA_CORTE = LOCAL / "corte.json"           # qué filas son validación y prueba (lo escribe el 01, lo leen el 04 y el 05)
TRATADO = RAIZ / "datos" / "tratado"

# ---------------------------------------------------------------- el caso
CASO = 3                                    # 1 y 3 = binario · 2 = multiclase
OBJETIVO = "binary:logistic" if CASO in (1, 3) else "multi:softprob"   # en la NAM: sigmoide + entropía cruzada binaria
PREFIJO_BUCKET = "ity1102-ep3"
PREFIJO_S3 = "fraude-nam"
NOMBRE_ENDPOINT = "ity1102-ep3-nam"
VARIANTE = "AllTraffic"

# Versiones de los contenedores de TensorFlow de SageMaker (Keras 3, como en el proyecto)
TF_ENTRENAMIENTO = "2.16.2"
TF_INFERENCIA = "2.16.1"


# ---------------------------------------------------------------- estado compartido entre scripts
def leer_estado():
    return json.loads(RUTA_ESTADO.read_text()) if RUTA_ESTADO.exists() else {}


def guardar_estado(**cambios):
    est = leer_estado()
    est.update(cambios)
    RUTA_ESTADO.write_text(json.dumps(est, indent=2, ensure_ascii=False))
    return est


def requerir(est, clave, script_previo):
    if not est.get(clave):
        sys.exit(f"ERROR: falta '{clave}' en {RUTA_ESTADO.name}. Ejecuta antes {script_previo}.")
    return est[clave]


def marca_tiempo():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------- AWS
def sesion():
    import boto3
    s = boto3.Session()
    return boto3.Session(region_name=s.region_name or os.getenv("AWS_REGION") or "us-east-1")


def id_cuenta(sess):
    return sess.client("sts").get_caller_identity()["Account"]


def rol_ejecucion(sess, rol_arg=None):
    """ARN del rol que usará SageMaker: --rol, luego SAGEMAKER_ROLE_ARN, luego 'LabRole' (AWS Academy)."""
    if rol_arg:
        return rol_arg
    if os.getenv("SAGEMAKER_ROLE_ARN"):
        return os.environ["SAGEMAKER_ROLE_ARN"]
    from botocore.exceptions import ClientError
    iam = sess.client("iam")
    try:
        return iam.get_role(RoleName="LabRole")["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            sys.exit("ERROR: no encontré el rol 'LabRole'. Indica el rol de SageMaker con --rol <ARN> "
                     "o con la variable SAGEMAKER_ROLE_ARN.")
        # sin permiso para iam:GetRole (típico en laboratorios): se asume el ARN estándar de AWS Academy
        arn = f"arn:aws:iam::{id_cuenta(sess)}:role/LabRole"
        print(f"   (sin permiso para consultar IAM; asumo el rol {arn})")
        return arn


def uri_imagen(region, alcance, override=None):
    """URI del contenedor de TensorFlow de SageMaker. alcance = 'training' | 'inference'."""
    if override:
        return override
    version = TF_ENTRENAMIENTO if alcance == "training" else TF_INFERENCIA
    try:
        from sagemaker import image_uris
        return image_uris.retrieve(framework="tensorflow", region=region, version=version, py_version="py310",
                                   image_scope=alcance, instance_type="ml.m5.xlarge")
    except Exception as e:      # SDK ausente o versión sin ese contenedor: patrón estándar de los Deep Learning Containers
        cuenta = "763104351884"
        host = f"{cuenta}.dkr.ecr.{region}.amazonaws.com"
        if alcance == "training":
            uri = f"{host}/tensorflow-training:{version}-cpu-py310-ubuntu20.04-sagemaker"
        else:
            uri = f"{host}/tensorflow-inference:{version}-cpu"
        print(f"   AVISO: no pude resolver la imagen con el SDK de SageMaker ({type(e).__name__}); uso {uri}.\n"
              f"          Si falla, pásala a mano con --imagen <URI> (lista: docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths).")
        return uri


# ---------------------------------------------------------------- datos (los usan 01 y 04)
def importar_entrenar():
    """Reutiliza servicio-ia/entrenar.py: una sola fuente de verdad para preprocesador, factores y métricas."""
    sys.path.insert(0, str(RAIZ / "servicio-ia"))
    import entrenar
    return entrenar


def cargar_tratado(tratado=TRATADO):
    """Lee variables_modelo.csv igual que entrenar.py y devuelve (d, NUM, CAT, FEATURES)."""
    var = json.loads((Path(tratado) / "variables.json").read_text())
    NUM, CAT = var["numericas"], var["categoricas"]
    d = pd.read_csv(Path(tratado) / "variables_modelo.csv", parse_dates=["ts_utc"], dtype={c: str for c in CAT})
    for c in CAT:                                   # categorías como texto; el nulo es NaN (igual que la API)
        d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
    return d, NUM, CAT, NUM + CAT


def separar(d):
    tr, va, te = (d[d["split"] == s].copy() for s in ("train", "val", "test"))
    if min(len(tr), len(va), len(te)) == 0:
        sys.exit("ERROR: algún conjunto (train/val/test) quedó vacío en variables_modelo.csv")
    return tr, va, te


# ---------------------------------------------------------------- corte train / validación / prueba
SEMILLA_CORTE = 42


def separar_corte(d, corte="estratificado", semilla=SEMILLA_CORTE):
    """'estratificado' (por defecto): la PRUEBA se aparta ANTES de entrenar, al azar y estratificada por y
         (train_test_split(test_size=0.2, stratify=y, random_state=42)); el 80 % restante se divide 75/25 en entrenamiento
         y validación, también estratificado. Resultado 60/20/20 con la misma proporción de fraude en cada parte.
       'fecha': el corte temporal de EP2 (columna split de preparar.py)."""
    if corte == "fecha":
        return separar(d)
    from sklearn.model_selection import train_test_split
    resto, te = train_test_split(d, test_size=0.2, stratify=d["y"], random_state=semilla)
    tr, va = train_test_split(resto, test_size=0.25, stratify=resto["y"], random_state=semilla)
    return tr.copy(), va.copy(), te.copy()


def guardar_corte(corte, tr, va, te, semilla=SEMILLA_CORTE):
    LOCAL.mkdir(parents=True, exist_ok=True)
    RUTA_CORTE.write_text(json.dumps({
        "corte": corte, "semilla": semilla if corte == "estratificado" else None,
        "filas": {"train": len(tr), "validacion": len(va), "prueba": len(te)},
        "fraudes": {"train": int(tr["y"].sum()), "validacion": int(va["y"].sum()), "prueba": int(te["y"].sum())},
        "ids_validacion": va["transaccion_id"].tolist(), "ids_prueba": te["transaccion_id"].tolist()}))


def cargar_corte(d):
    """Reconstruye (corte, train, validación, prueba) tal como los dejó el paso 01."""
    if not RUTA_CORTE.exists():
        sys.exit("ERROR: falta aws/local/corte.json. Ejecuta antes 01_s3_subir_datos.py.")
    c = json.loads(RUTA_CORTE.read_text())
    ids_te, ids_va = set(c["ids_prueba"]), set(c["ids_validacion"])
    te, va = d[d["transaccion_id"].isin(ids_te)].copy(), d[d["transaccion_id"].isin(ids_va)].copy()
    tr = d[~d["transaccion_id"].isin(ids_te | ids_va)].copy()
    if len(te) != len(ids_te) or len(va) != len(ids_va):
        sys.exit("ERROR: los datos de datos/tratado cambiaron desde el paso 01 (faltan ids del corte). Vuelve a ejecutarlo.")
    return c["corte"], tr, va, te


def importar_script(nombre):
    """Importa otro script numerado de esta carpeta (p. ej. '04_invocar_y_metricas') para reutilizar sus funciones."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(nombre, AQUI / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo
