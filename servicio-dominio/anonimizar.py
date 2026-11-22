"""anonimizar.py · Quita los datos personales antes de que un texto salga hacia la API de NVIDIA (EP3, sección 2.3).

Dos defensas, en este orden:
  1. Lista blanca (mensaje_para_llm): el mensaje que va a la cola se arma SOLO con campos permitidos.
     Nombre, RUT, correo, teléfono, dirección, tarjeta, usuario_id, IP y dispositivo nunca se agregan.
  2. Expresiones regulares (anonimizar): si algún dato personal viene escondido en un texto libre
     (por ejemplo en el motivo de la lista negra), se reemplaza por una etiqueta antes de enviarlo.
"""
from __future__ import annotations

import re
from typing import Any

# El orden importa: la tarjeta va antes que el teléfono (un PAN contiene secuencias que parecen teléfonos)
PATRONES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[CORREO]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[TARJETA]"),
    (re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dkK]\b"), "[RUT]"),
    (re.compile(r"(?:\+?56[ -]?)?\b9[ -]?\d{4}[ -]?\d{4}\b"), "[TELEFONO]"),
    (re.compile(r"(?:\+?56[ -]?)?\b2[ -]?\d{4}[ -]?\d{4}\b"), "[TELEFONO]"),
]

# Campos que SÍ pueden llegar al LLM: lo necesario para explicar la decisión, nada que identifique a la persona
CAMPOS_PERMITIDOS = ("prediccion_id", "decision", "probabilidad_fraude", "monto_usd", "moneda", "canal",
                     "pais_transaccion", "mcc", "origen", "razones")
CAMPOS_RAZON = ("tipo", "factor", "descripcion", "efecto", "aporte_logodds")


def anonimizar(texto: str) -> str:
    """Reemplaza RUT, correo, teléfono y número de tarjeta por una etiqueta."""
    if not texto:
        return texto
    for patron, etiqueta in PATRONES:
        texto = patron.sub(etiqueta, texto)
    return texto


def _limpiar(v: Any) -> Any:
    return anonimizar(v) if isinstance(v, str) else v


def mensaje_para_llm(prediccion: dict) -> dict:
    """Arma el mensaje de la cola con lista blanca de campos y textos anonimizados."""
    msg = {k: _limpiar(prediccion[k]) for k in CAMPOS_PERMITIDOS if k in prediccion and k != "razones"}
    msg["razones"] = [{k: _limpiar(r[k]) for k in CAMPOS_RAZON if k in r} for r in prediccion.get("razones", [])]
    return msg
