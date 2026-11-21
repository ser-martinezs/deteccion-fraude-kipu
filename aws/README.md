# EP3 · Caso 3 (fraude) · NAM en SageMaker

Copia esta carpeta como `aws/` en la raíz del repositorio (junto a `datos/` y `servicio-ia/`) y ejecuta en orden:

```bash
pip install -r aws/requirements-aws.txt
python aws/01_s3_subir_datos.py        # S3 → ity1102-ep3-<cuenta>   (train y validation; la prueba NO se sube)
python aws/02_entrenar_nam.py          # SageMaker → Training jobs   (NAM con TensorFlow, ~5-10 min)
python aws/03_desplegar_endpoint.py    # SageMaker → Inference → Endpoints (crea o actualiza, mismo nombre)
python aws/04_invocar_y_metricas.py    # invoca con la prueba y calcula las métricas; CloudWatch
```

Los scripts se pasan datos entre sí mediante `aws/estado.json`. Necesitas credenciales de AWS configuradas
(en AWS Academy: las del Learner Lab) y un rol de SageMaker (por defecto `LabRole`; otro con `--rol <ARN>`).

| Script | Parámetros útiles |
|---|---|
| 01 | `--bucket`, `--sufijo` |
| 02 | `--epocas`, `--lr`, `--batch`, `--l2`, `--capas 16,8`, `--peso-fraude`, `--instancia`, `--imagen`, `--no-esperar` |
| 03 | `--endpoint`, `--memoria` (1024-6144 MB), `--concurrencia` (1-200), `--limpiar`, `--imagen` |
| 04 | `--lote`, `--sin-cloudwatch`, `--espera-cw` |

## Contrato del endpoint (`src/inference.py`)

Entrada (`application/json`): las 21 variables de `datos/tratado/variables.json` (15 numéricas + 6 categóricas); `null` o ausente = faltante.

```json
{"variables": {"log_monto_usd": 3.78, "hora_local": 1, "es_madrugada": 1, "...": "...", "canal": "web", "mcc": "5411"}}
```

Salida:

```json
{"probabilidad": 0.0098, "aportes": {"pais_distinto_usuario": 3.03, "es_madrugada": 2.09, "...": 0.0}, "version_modelo": "3.0.0"}
```

- `probabilidad`: de fraude **en producción** (ya calibrada con la validación dentro del entrenamiento).
- `aportes`: log-odds de cada factor frente a la transacción promedio de entrenamiento (positivo = sube el riesgo). `servicio-ia` solo ordena por valor absoluto y toma los 3 primeros.
- Varias transacciones a la vez (lo usa el script 04): `{"instancias": [{...}, {...}]}` → `{"resultados": [{probabilidad, aportes}, ...], "version_modelo": ...}`.
- Los umbrales **no** viven en el endpoint: el 04 deja `aws/umbrales_sagemaker.json` (mismo formato que `servicio-ia/model/umbrales.json`) para la configuración de `servicio-ia`.

## Qué cambia respecto a XGBoost

- **Entrenamiento:** la NAM no es un algoritmo integrado, así que corre en el contenedor oficial de TensorFlow 2.16
  (script mode, `src/train_nam.py`). `binary:logistic` se traduce en salida sigmoide + entropía cruzada binaria.
- **Preprocesamiento:** el paso 01 ajusta el preprocesador de `servicio-ia/entrenar.py` solo con entrenamiento y lo exporta a
  JSON; `inference.py` lo aplica con numpy dentro del endpoint (idéntico al de scikit-learn, verificado en las 28.040 filas).
- **model.tar.gz:** `1/` (SavedModel con salidas `prob` y `aportes`), `code/inference.py`, `metadata.json`, `preprocesador.json`.
- **Métricas del paso 04:** reutilizan `entrenar.py` (umbrales, costo), comparables con `servicio-ia/reporte_entrenamiento.json`.
  Los umbrales se buscan con la validación, nunca con la prueba (que no se sube a S3).
- **Endpoint serverless:** "memoria y concurrencia" son `MemorySizeInMB` y `MaxConcurrency`. La primera invocación
  puede tardar por el arranque en frío (el script reintenta).
