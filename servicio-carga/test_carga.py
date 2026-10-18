"""Prueba del servicio de carga: correrlo dos veces NO duplica filas.

Requiere DATABASE_URL (en CI, un PostgreSQL de servicio) y datos/tratado (lo genera datos/preparar.py).
"""
import os
from pathlib import Path

import psycopg
import pytest

import carga

TRATADO = Path(__file__).resolve().parent.parent / "datos" / "tratado"
pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL") or not TRATADO.exists(),
                                reason="sin base de datos o sin datos/tratado")


def conteos():
    with psycopg.connect(os.environ["DATABASE_URL"]) as c:
        return {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ["usuarios", "dispositivos", "lista_negra", "transacciones"]}


def test_carga_dos_veces_no_duplica(monkeypatch):
    monkeypatch.setattr(carga, "DIR_DATOS", TRATADO)
    carga.main()
    primera = conteos()
    carga.main()
    assert conteos() == primera
    assert primera["usuarios"] == 5000 and primera["transacciones"] > 27000
