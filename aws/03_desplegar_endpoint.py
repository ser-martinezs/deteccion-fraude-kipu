#!/usr/bin/env python3
"""
03_desplegar_endpoint.py · Crea el modelo, la configuración (memoria y concurrencia) y el endpoint.
                           Si el endpoint ya existe, lo ACTUALIZA sin cambiarle el nombre.
                           →  consola: SageMaker → Inference → Endpoints

Es un endpoint serverless (la configuración se expresa como memoria y concurrencia máxima):
  --memoria       1024, 2048, 3072, 4096, 5120 o 6144 MB      (por defecto 2048)
  --concurrencia  1 a 200 invocaciones simultáneas             (por defecto 5)

Los modelos y las configuraciones de endpoint de SageMaker son inmutables: cada ejecución crea un modelo y una
configuración NUEVOS (con sello de tiempo) y el endpoint, que conserva su nombre, pasa a usarlos (UpdateEndpoint,
sin caída de servicio). Los anteriores quedan en la consola; --limpiar borra los de la versión previa.

Uso:  python aws/03_desplegar_endpoint.py [--endpoint NOMBRE] [--memoria 2048] [--concurrencia 5] [--limpiar]
"""
import argparse
import sys

import config

MEMORIAS_VALIDAS = (1024, 2048, 3072, 4096, 5120, 6144)


def endpoint_existe(sm, nombre):
    from botocore.exceptions import ClientError
    try:
        return sm.describe_endpoint(EndpointName=nombre)
    except ClientError as e:
        if "Could not find endpoint" in e.response["Error"].get("Message", ""):
            return None
        raise


def esperar_endpoint(sm, nombre):
    from botocore.exceptions import WaiterError
    try:
        sm.get_waiter("endpoint_in_service").wait(EndpointName=nombre, WaiterConfig={"Delay": 15, "MaxAttempts": 80})
    except WaiterError:
        d = sm.describe_endpoint(EndpointName=nombre)
        sys.exit(f"ERROR: el endpoint quedó en {d['EndpointStatus']}: {d.get('FailureReason', '(sin detalle)')}\n"
                 f"Detalle en CloudWatch → Logs → /aws/sagemaker/Endpoints/{nombre}")


def borrar_endpoint_fallido(sm, nombre):
    print(f"3/3 El endpoint {nombre} está en estado Failed: lo borro antes de recrearlo")
    sm.delete_endpoint(EndpointName=nombre)
    sm.get_waiter("endpoint_deleted").wait(
        EndpointName=nombre,
        WaiterConfig={"Delay": 10, "MaxAttempts": 60},
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=config.NOMBRE_ENDPOINT, help="nombre del endpoint (no cambia entre ejecuciones)")
    ap.add_argument("--memoria", type=int, default=2048, choices=MEMORIAS_VALIDAS, help="MB de memoria del endpoint serverless")
    ap.add_argument("--concurrencia", type=int, default=5, help="concurrencia máxima (1-200)")
    ap.add_argument("--rol", help="ARN del rol de SageMaker (por defecto el del paso 02 o LabRole)")
    ap.add_argument("--imagen", help="URI del contenedor de inferencia (por defecto TensorFlow Serving 2.16 oficial)")
    ap.add_argument("--limpiar", action="store_true", help="tras actualizar, borra el modelo y la configuración anteriores")
    args = ap.parse_args(argv)
    if not 1 <= args.concurrencia <= 200:
        sys.exit("ERROR: --concurrencia debe estar entre 1 y 200")

    est = config.leer_estado()
    artefacto = config.requerir(est, "model_artifact", "02_entrenar_nam.py")
    sess = config.sesion()
    region = sess.region_name
    sm = sess.client("sagemaker")
    rol = args.rol or est.get("rol_arn") or config.rol_ejecucion(sess)
    imagen = config.uri_imagen(region, "inference", args.imagen)

    sello = config.marca_tiempo()
    modelo = f"{args.endpoint}-modelo-{sello}"
    cfg = f"{args.endpoint}-cfg-{sello}"

    print(f"1/3 Modelo: {modelo}\n    imagen: {imagen}\n    artefacto: {artefacto}")
    sm.create_model(ModelName=modelo, ExecutionRoleArn=rol,
                    PrimaryContainer={"Image": imagen, "ModelDataUrl": artefacto, "Mode": "SingleModel"},
                    Tags=[{"Key": "proyecto", "Value": "ity1102-ep3"}])

    print(f"2/3 Configuración: {cfg} · memoria {args.memoria} MB · concurrencia máxima {args.concurrencia}")
    sm.create_endpoint_config(
        EndpointConfigName=cfg,
        ProductionVariants=[{"VariantName": config.VARIANTE, "ModelName": modelo,
                             "ServerlessConfig": {"MemorySizeInMB": args.memoria, "MaxConcurrency": args.concurrencia}}],
        Tags=[{"Key": "proyecto", "Value": "ity1102-ep3"}])

    previo = endpoint_existe(sm, args.endpoint)
    anterior_cfg = previo["EndpointConfigName"] if previo else None
    if previo is None:
        print(f"3/3 El endpoint {args.endpoint} no existe: lo creo")
        sm.create_endpoint(EndpointName=args.endpoint, EndpointConfigName=cfg, Tags=[{"Key": "proyecto", "Value": "ity1102-ep3"}])
        accion = "creado"
    else:
        if previo["EndpointStatus"] == "Failed":
            borrar_endpoint_fallido(sm, args.endpoint)
            print(f"3/3 El endpoint {args.endpoint} fue eliminado: lo recreo")
            sm.create_endpoint(EndpointName=args.endpoint, EndpointConfigName=cfg, Tags=[{"Key": "proyecto", "Value": "ity1102-ep3"}])
            accion = "recreado"
        elif previo["EndpointStatus"] in ("Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting"):
            print("    esperando a que termine su operación en curso...")
            esperar_endpoint(sm, args.endpoint)
            sm.update_endpoint(EndpointName=args.endpoint, EndpointConfigName=cfg)
            accion = "actualizado"
        else:
            print(f"3/3 El endpoint {args.endpoint} ya existe ({previo['EndpointStatus']}): lo actualizo conservando su nombre")
            sm.update_endpoint(EndpointName=args.endpoint, EndpointConfigName=cfg)
            accion = "actualizado"

    print("    esperando a InService (puede tardar unos minutos)...")
    esperar_endpoint(sm, args.endpoint)
    config.guardar_estado(endpoint=args.endpoint, modelo=modelo, configuracion_endpoint=cfg,
                          memoria_mb=args.memoria, concurrencia=args.concurrencia, rol_arn=rol)
    print(f"\nEndpoint {args.endpoint} {accion} y en servicio. Consola: SageMaker → Inference → Endpoints")

    if args.limpiar and anterior_cfg and anterior_cfg != cfg:
        try:
            modelo_previo = sm.describe_endpoint_config(EndpointConfigName=anterior_cfg)["ProductionVariants"][0]["ModelName"]
            sm.delete_endpoint_config(EndpointConfigName=anterior_cfg)
            sm.delete_model(ModelName=modelo_previo)
            print(f"Borrados la configuración {anterior_cfg} y el modelo {modelo_previo} de la versión anterior")
        except Exception as e:
            print(f"(no pude limpiar la versión anterior: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
