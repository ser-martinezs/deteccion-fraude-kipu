import pandas as pd
import pytest
from pathlib import Path

RUTA_TRATADO = Path(__file__).resolve().parent / "tratado"     # no depende del directorio desde el que se ejecute pytest


@pytest.fixture(scope="module")
def tx():
    return pd.read_csv(RUTA_TRATADO / "transacciones.csv")


def test_transacciones_sin_duplicados_clave(tx):
    # Regla R01: no debe haber transaccion_id duplicadas
    assert tx["transaccion_id"].duplicated().sum() == 0, "Existen transaccion_id duplicadas"


def test_transacciones_sin_nulos_en_claves(tx):
    # Claves primarias y campos críticos no pueden ser nulos
    assert tx["transaccion_id"].notna().all(), "Hay nulos en transaccion_id"
    assert tx["ts_utc"].notna().all(), "Hay nulos en timestamp UTC"
    assert tx["monto_usd"].notna().all(), "Hay nulos en monto_usd"


def test_usuarios_y_dispositivos_claves_unicas():
    usr = pd.read_csv(RUTA_TRATADO / "usuarios.csv")
    dsp = pd.read_csv(RUTA_TRATADO / "dispositivos.csv")
    assert usr["usuario_id"].duplicated().sum() == 0, "Hay usuario_id duplicados"
    assert dsp["dispositivo_id"].duplicated().sum() == 0, "Hay dispositivo_id duplicados"


def test_valores_dentro_de_rango(tx):
    assert (tx["monto_usd"] >= 0).all(), "monto_usd negativo (R08 debía llevarlo a valor absoluto)"
    assert set(tx["moneda"].unique()) <= {"CLP", "USD"}, "moneda fuera de CLP/USD (R07)"
    assert tx["canal"].isin(["app", "web", "pos", "api"]).all(), "canal sin normalizar (R03)"
    assert pd.to_datetime(tx["ts_utc"], errors="coerce").notna().all(), "ts_utc no se puede interpretar (R04)"


def test_variables_modelo_con_split_valido():
    vm = pd.read_csv(RUTA_TRATADO / "variables_modelo.csv", usecols=["transaccion_id", "ts_utc", "split", "y"])
    assert vm["transaccion_id"].duplicated().sum() == 0
    assert set(vm["split"].unique()) == {"train", "val", "test"}
    ts = pd.to_datetime(vm["ts_utc"])
    # la separación es por fecha: lo que se entrena es anterior a lo que se valida y se prueba
    assert ts[vm["split"] == "train"].max() <= ts[vm["split"] == "val"].min()
    assert ts[vm["split"] == "val"].max() <= ts[vm["split"] == "test"].min()
    for s in ("train", "val", "test"):
        assert vm.loc[vm["split"] == s, "y"].sum() > 0, f"el conjunto {s} no tiene fraudes"
