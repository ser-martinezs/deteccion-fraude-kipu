"""
Paridad entrenamiento/servicio: para transacciones reales del histórico, las variables que calcula la API
(a partir del dato CRUDO + perfil de usuario/dispositivo) deben ser IGUALES a las de variables_modelo.csv,
que son las que vio el modelo al entrenar. Si este test falla, el modelo en línea ve otra cosa que la que aprendió.

Requiere datos/crudo y datos/tratado (en CI se genera con preparar.py antes de pytest).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as main_module

RAIZ = Path(__file__).resolve().parent.parent
CRUDO, TRATADO = RAIZ / "datos" / "crudo", RAIZ / "datos" / "tratado"
MUESTRA = 300

pytestmark = pytest.mark.skipif(not (TRATADO / "variables_modelo.csv").exists() or not (CRUDO / "transacciones.csv").exists(),
                                reason="faltan datos/crudo o datos/tratado (corre datos/preparar.py)")


def _none(v):
    return None if pd.isna(v) else v


def _perfil(df, clave, columnas):
    if clave is None or clave not in df.index:
        return None
    fila = df.loc[clave]
    return {c: _none(fila[c]) for c in columnas}


def test_variables_del_servicio_coinciden_con_las_del_entrenamiento():
    var = json.loads((TRATADO / "variables.json").read_text(encoding="utf-8"))
    num, cat = var["numericas"], var["categoricas"]
    vm = pd.read_csv(TRATADO / "variables_modelo.csv", dtype={c: str for c in cat}).set_index("transaccion_id")
    raw = pd.read_csv(CRUDO / "transacciones.csv", dtype=str).drop_duplicates("transaccion_id").set_index("transaccion_id")
    usr = pd.read_csv(TRATADO / "usuarios.csv").drop_duplicates("usuario_id").set_index("usuario_id")
    dsp = pd.read_csv(TRATADO / "dispositivos.csv").drop_duplicates("dispositivo_id").set_index("dispositivo_id")

    ids = vm.sample(min(MUESTRA, len(vm)), random_state=0).index
    distintas = []
    for tid in ids:
        r = raw.loc[tid]
        did = _none(r["dispositivo_id"])
        payload = {
            # El CSV contiene comercios faltantes; el servicio no usa este identificador
            # para construir variables cuando la lista negra ya fue resuelta.
            "transaccion_id": tid, "usuario_id": r["usuario_id"], "comercio_id": _none(r["comercio_id"]) or "", "dispositivo_id": did,
            "timestamp": r["timestamp"], "monto": r["monto"], "moneda": _none(r["moneda"]), "canal": r["canal"],
            "pais_transaccion": r["pais_transaccion"], "mcc": r["mcc"], "latitud": _none(r["latitud"]), "longitud": _none(r["longitud"]),
            "ip": _none(r["ip"]), "tarjeta_enmascarada": _none(r["tarjeta_enmascarada"]),
            "intentos_previos_1h": _none(r["intentos_previos_1h"]),
            "usuario": _perfil(usr, r["usuario_id"], ["pais_usuario", "segmento", "verificado", "telefono_verificado", "nivel_kyc", "dominio_desechable", "fecha_registro"]),
            "dispositivo": _perfil(dsp, did, ["tipo_dispositivo", "pais_disp", "emulador"]),
            # la lista negra la cruza el dominio contra la base; aquí se toma la del entrenamiento
            "lista_negra_interna": bool(vm.loc[tid, "lista_negra_interna"] == 1),
            "lista_negra_dura": bool(vm.loc[tid, "lista_negra_dura"]),
        }
        x = main_module.construir_features(main_module.TransaccionInput(**payload)).iloc[0]
        esperado = vm.loc[tid]
        for c in num:
            a, b = float(x[c]), float(esperado[c])
            if not (np.isclose(a, b, rtol=1e-6, atol=1e-9) or (np.isnan(a) and np.isnan(b))):
                distintas.append((tid, c, a, b))
        for c in cat:
            a, b = x[c], esperado[c]
            if not (a == b or (pd.isna(a) and pd.isna(b))):
                distintas.append((tid, c, a, b))
    assert not distintas, f"{len(distintas)} variables difieren entre API y entrenamiento, p. ej.: {distintas[:5]}"
