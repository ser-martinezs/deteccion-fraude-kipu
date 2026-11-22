"""Prueba de carga de EP3 (sección 4.3) contra el sistema en AWS.

Simula al core de pagos pidiendo decisiones a POST /api/predict con la clave de API. Las transacciones son de montos
bajos en CLP y comercio habitual, para que casi todas salgan "aprobar" y la prueba mida el camino síncrono
(gateway -> dominio -> SageMaker) sin llenar la cola del LLM, que tiene su propio límite por minuto.

Antes de correrla, en el .env de la EC2: LIMITE_POR_MINUTO=100000 y `docker compose ... up -d dominio`
(si no, el límite por cliente responde 429). Al terminar, volverlo a 3000.

    pip install locust
    API_KEY=<clave> locust -f pruebas/locustfile.py --headless -u 20 -r 5 -t 3m \
        --host https://<ip-con-guiones>.sslip.io --csv pruebas/resultados/carga

Del archivo carga_stats.csv salen p50, p95, p99, solicitudes por segundo y tasa de errores (fila "Aggregated").
Para la defensa (IE11), la versión de 1 minuto: -t 1m.
"""
import os
import random

from locust import HttpUser, between, task

CLAVE = os.environ.get("API_KEY", "")
USUARIOS = [("KP301626", "CM40294"), ("KP300267", "CM40847"), ("KP300015", "CM40294"), ("KP300102", "CM40311")]


def transaccion() -> dict:
    usuario, comercio = random.choice(USUARIOS)
    return {"usuario_id": usuario, "comercio_id": comercio, "monto": random.choice([4500, 8500, 12000, 18500, 25000]),
            "moneda": "CLP", "canal": random.choice(["pos", "app"]), "mcc": random.choice(["5411", "5812", "5912"]),
            "pais_transaccion": "CL", "intentos_previos_1h": 0}


class CorePagos(HttpUser):
    wait_time = between(0.5, 1.0)

    def on_start(self):
        if not CLAVE:
            raise RuntimeError("Falta la variable API_KEY con la clave de API del sistema")

    @task(10)
    def predecir(self):
        with self.client.post("/api/predict", json=transaccion(), headers={"X-API-Key": CLAVE},
                              name="POST /api/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:120]}")
            elif r.json().get("origen") == "reserva":
                # respondió, pero por la política de reserva (SageMaker no alcanzó): se cuenta aparte para el informe
                r.success()
                self.environment.events.request.fire(request_type="RESERVA", name="decisiones por reserva",
                                                     response_time=r.elapsed.total_seconds() * 1000,
                                                     response_length=0, exception=None, context={})

    @task(1)
    def salud(self):
        self.client.get("/api/health", name="GET /api/health")
