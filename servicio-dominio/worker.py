"""worker.py · Worker del LLM (EP3, sección 2.3). Corre con la misma imagen que el dominio: `python worker.py`.

Consume la cola `resumenes_llm` de RabbitMQ, pide a la API de NVIDIA un resumen en lenguaje común para el analista
de la cola de revisión (RF5, RNF6) y lo guarda en predicciones.resumen_llm.

El sistema sobrevive al LLM:
  * tiempo máximo de espera por llamada (NVIDIA_TIMEOUT_S),
  * reintentos con espera creciente ante 429, timeout o caída (1, 2, 4, 8 s + azar), hasta NVIDIA_REINTENTOS,
  * clave inválida (401/403): no se reintenta,
  * si no se logra, la predicción queda en modo degradado: el analista ve la decisión y las razones del modelo,
    sin el texto del LLM. El mensaje se confirma (ack) igual, para que no quede dando vueltas en la cola.
El LLM nunca decide: solo redacta a partir de las razones que ya calculó el modelo.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time

import pika
import psycopg
from openai import (APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI,
                    PermissionDeniedError, RateLimitError)

from anonimizar import anonimizar

DSN = os.getenv("DATABASE_URL", "postgresql://kipu:kipu@db:5432/kipu")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/%2F")
COLA_LLM = os.getenv("COLA_LLM", "resumenes_llm")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "")
NVIDIA_TIMEOUT_S = float(os.getenv("NVIDIA_TIMEOUT_S", "15"))
NVIDIA_REINTENTOS = int(os.getenv("NVIDIA_REINTENTOS", "4"))
PAUSA_MIN_S = float(os.getenv("NVIDIA_PAUSA_MIN_S", "1.5"))   # ~40 solicitudes/min por clave: no pasar de ~40

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("worker-llm")

PROMPT = ("Eres asistente de un analista antifraude de una billetera digital chilena. Recibes la decisión del modelo "
          "(aprobar, revisar o bloquear), la probabilidad de fraude y las razones ordenadas por peso. Escribe en español "
          "simple, en un máximo de 3 frases, por qué la transacción quedó en esa decisión y qué debería revisar primero "
          "el analista. No inventes datos que no estén en el mensaje y no cambies la decisión.")

_cliente: OpenAI | None = None


def cliente() -> OpenAI:
    global _cliente
    if _cliente is None:
        # max_retries=0: los reintentos los controlamos nosotros (y quedan en el log)
        _cliente = OpenAI(base_url=NVIDIA_BASE_URL, api_key=os.getenv("NVIDIA_API_KEY", "sin-clave"),
                          timeout=NVIDIA_TIMEOUT_S, max_retries=0)
    return _cliente


def espera(intento: int) -> float:
    return 2 ** intento + random.random()


def resumir(mensaje: dict, dormir=time.sleep) -> tuple[str, str | None, str]:
    """Devuelve (estado, resumen, detalle). estado = listo | degradado. Nunca lanza excepción."""
    texto = anonimizar(json.dumps(mensaje, ensure_ascii=False))       # segunda defensa: nada personal sale hacia NVIDIA
    ultimo = "sin intentos"
    for intento in range(NVIDIA_REINTENTOS):
        try:
            r = cliente().chat.completions.create(
                model=NVIDIA_MODEL, temperature=0.2, max_tokens=220,
                messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": texto}])
            return "listo", r.choices[0].message.content.strip(), "ok"
        except (AuthenticationError, PermissionDeniedError) as e:
            log.error("llm_clave_invalida prediccion_id=%s status=%s: no se reintenta", mensaje.get("prediccion_id"), e.status_code)
            return "degradado", None, f"clave inválida ({e.status_code})"
        except RateLimitError:
            ultimo = "429 Too Many Requests"
        except APITimeoutError:
            ultimo = f"timeout de {NVIDIA_TIMEOUT_S:.0f} s"
        except APIConnectionError:
            ultimo = "API de NVIDIA no disponible"
        except APIStatusError as e:
            if e.status_code < 500:
                log.error("llm_error prediccion_id=%s status=%s: no se reintenta", mensaje.get("prediccion_id"), e.status_code)
                return "degradado", None, f"error {e.status_code}"
            ultimo = f"error {e.status_code} de NVIDIA"
        if intento < NVIDIA_REINTENTOS - 1:
            s = espera(intento)
            log.warning("llm_reintento prediccion_id=%s intento=%d motivo=%s espera_s=%.1f",
                        mensaje.get("prediccion_id"), intento + 1, ultimo, s)
            dormir(s)
    log.error("llm_degradado prediccion_id=%s motivo=%s tras %d intentos", mensaje.get("prediccion_id"), ultimo, NVIDIA_REINTENTOS)
    return "degradado", None, ultimo


def guardar(con: psycopg.Connection, prediccion_id: int, estado: str, resumen: str | None, detalle: str) -> None:
    con.execute("UPDATE predicciones SET estado_llm = %s, resumen_llm = %s, llm_actualizado = now() WHERE prediccion_id = %s",
                (estado, resumen, prediccion_id))
    con.execute("""INSERT INTO estado_servicios (servicio, estado, detalle, actualizado) VALUES ('llm', %s, %s, now())
                   ON CONFLICT (servicio) DO UPDATE SET estado = EXCLUDED.estado, detalle = EXCLUDED.detalle,
                                                        actualizado = EXCLUDED.actualizado""",
                ("ok" if estado == "listo" else "degradado", detalle))
    con.commit()


def procesar(con: psycopg.Connection, cuerpo: bytes) -> str:
    mensaje = json.loads(cuerpo)
    t0 = time.perf_counter()
    estado, resumen, detalle = resumir(mensaje)
    guardar(con, int(mensaje["prediccion_id"]), estado, resumen, detalle)
    log.info("llm_resultado prediccion_id=%s estado=%s latencia_ms=%.0f", mensaje["prediccion_id"], estado,
             (time.perf_counter() - t0) * 1000)
    return estado


def main() -> None:
    if not NVIDIA_MODEL:
        log.warning("NVIDIA_MODEL vacío: todas las llamadas fallarán y las predicciones quedarán en modo degradado")
    while True:   # si RabbitMQ o la base se caen, el worker espera y se reconecta: nunca bota el sistema
        try:
            with psycopg.connect(DSN) as con:
                conexion = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
                canal = conexion.channel()
                canal.queue_declare(queue=COLA_LLM, durable=True)
                canal.basic_qos(prefetch_count=1)      # de a un mensaje: respeta el límite por minuto de NVIDIA
                log.info("worker conectado a la cola %s (modelo %s)", COLA_LLM, NVIDIA_MODEL or "-")
                for metodo, _props, cuerpo in canal.consume(COLA_LLM, inactivity_timeout=30):
                    if metodo is None:
                        continue
                    t0 = time.monotonic()
                    try:
                        procesar(con, cuerpo)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        log.error("mensaje_invalido descartado: %s", e.__class__.__name__)
                    canal.basic_ack(metodo.delivery_tag)
                    pausa = PAUSA_MIN_S - (time.monotonic() - t0)
                    if pausa > 0:
                        time.sleep(pausa)
        except (pika.exceptions.AMQPError, psycopg.OperationalError) as e:
            log.warning("worker sin conexión (%s): reintento en 5 s", e.__class__.__name__)
            time.sleep(5)


if __name__ == "__main__":
    main()
