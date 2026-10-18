"""servicio-carga · llena la base la primera vez que se levanta el sistema.

- Crea el esquema (CREATE ... IF NOT EXISTS).
- Copia cada CSV tratado (salida de datos/preparar.py) a una tabla temporal y hace
  INSERT ... ON CONFLICT DO NOTHING sobre la clave: si corre dos veces, NO duplica filas.
- Termina con código 0 (compose: restart "no"; el dominio espera service_completed_successfully).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import psycopg

DIR_DATOS = Path(os.getenv("DIR_DATOS", Path(__file__).resolve().parent / "datos"))
ESQUEMA = Path(__file__).resolve().parent / "schema.sql"
DSN = os.getenv("DATABASE_URL", "postgresql://kipu:kipu@db:5432/kipu")

# (tabla, columnas de la clave). Orden: primero las tablas de referencia
TABLAS = [("usuarios", "usuario_id"), ("dispositivos", "dispositivo_id"),
          ("lista_negra", "tipo, valor, fecha_alta"), ("transacciones", "transaccion_id")]


def conectar(intentos: int = 30) -> psycopg.Connection:
    for i in range(intentos):
        try:
            return psycopg.connect(DSN)
        except psycopg.OperationalError as e:
            print(f"[carga] base no disponible ({i + 1}/{intentos}): {e}", flush=True)
            time.sleep(2)
    sys.exit("[carga] no se pudo conectar a la base")


def cargar_tabla(cur: psycopg.Cursor, tabla: str, clave: str) -> tuple[int, int]:
    archivo = DIR_DATOS / f"{tabla}.csv"
    with archivo.open(newline="", encoding="utf-8") as f:
        columnas = next(csv.reader(f))
    cols = ", ".join(columnas)
    cur.execute(f"CREATE TEMP TABLE stg_{tabla} (LIKE {tabla} INCLUDING DEFAULTS) ON COMMIT DROP")
    with archivo.open("rb") as f, cur.copy(f"COPY stg_{tabla} ({cols}) FROM STDIN WITH (FORMAT csv, HEADER true)") as cp:
        while bloque := f.read(1 << 16):
            cp.write(bloque)
    cur.execute(f"SELECT count(*) FROM stg_{tabla}")
    leidas = cur.fetchone()[0]
    cur.execute(f"INSERT INTO {tabla} ({cols}) SELECT {cols} FROM stg_{tabla} ON CONFLICT ({clave}) DO NOTHING")
    return leidas, cur.rowcount


def main() -> None:
    t0 = time.time()
    with conectar() as con, con.cursor() as cur:
        cur.execute(ESQUEMA.read_text(encoding="utf-8"))
        for tabla, clave in TABLAS:
            leidas, nuevas = cargar_tabla(cur, tabla, clave)
            print(f"[carga] {tabla:<14} leídas={leidas:>6}  insertadas={nuevas:>6}  ya existían={leidas - nuevas:>6}",
                  flush=True)
        con.commit()
    print(f"[carga] OK en {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
