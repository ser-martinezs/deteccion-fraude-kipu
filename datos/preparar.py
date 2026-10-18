#!/usr/bin/env python3
"""
datos/preparar.py · Tratamiento de datos de Kipu Pagos (Caso 3, detección de fraude)

Lo ejecuta el pipeline ANTES de entrenar. Entrada: los 4 CSV crudos. Salida: tablas tratadas + reporte.

    python datos/preparar.py --entrada datos/crudo --salida datos/tratado

Qué hace (los números son los de la corrida; todo queda escrito en reporte_tratamiento.md/.json):
  1. Perfilado        filas, columnas, tipos, nulos y duplicados de cada archivo, ANTES y DESPUÉS.
  2. Limpieza         una regla escrita por problema (REGLAS): corregir, descartar o marcar, y por qué.
  3. Integración      une por claves e informa qué registros quedan sin pareja y qué se hace con ellos.
  4. Variables        cuáles son, de qué columnas salen y cómo se calculan (VARIABLES).
  5. Separación       entrenamiento / validación / prueba POR FECHA, fijada aquí, antes de entrenar.
  6. Salidas          tablas tratadas (para el servicio de carga) + variables_modelo.csv (para entrenar.py).

Código de salida: 0 = ok; 2 = los datos no cumplen una validación (categoría nueva, columna ausente, etc.).
Las funciones normalizar_* y construir_variables son las MISMAS que usará el servicio para una transacción en línea.
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVOS = {"usuarios": "usuario_id", "transacciones": "transaccion_id", "dispositivos": "dispositivo_id", "lista_negra": None}

# ------------------------------------------------------------------ constantes del tratamiento
OFFSET_CHILE_H = 4        # las fechas sin zona y las dd/mm/yyyy están en hora de Chile, UTC-4 fijo (verificado con los datos, ver README)
TURNO = (8, 22)           # analistas de 08:00 a 22:00 hora de Chile
TASA_CLP_USD = 950        # SUPUESTO: USD 26 ≈ $24.700 CLP (sección 2 del caso)
UMBRAL_MONEDA = 500       # CLP nunca baja de 901 y USD nunca pasa de 313: 500 separa sin ambigüedad (se verifica en cada corrida)
VENTANA_DUPLICADO_S = 180 # pagos idénticos a menos de 180 s son repetidos
FRACCIONES_SPLIT = (0.60, 0.20)   # entrenamiento 60%, validación 20%, prueba el resto, por fecha
MIN_FRAUDES_POR_CONJUNTO = 10
DESECHABLES = {"mailinator.com", "guerrillamail.com", "tempmail.io", "10minutemail.com"}

BOOL = {"1": 1, "0": 0, "1.0": 1, "0.0": 0, "si": 1, "no": 0, "true": 1, "false": 0}
PAIS = {"cl": "CL", "chile": "CL", "ar": "AR", "argentina": "AR", "pe": "PE", "peru": "PE",
        "br": "BR", "brazil": "BR", "brasil": "BR", "co": "CO", "colombia": "CO",
        "ve": "VE", "venezuela": "VE", "us": "US", "usa": "US", "eeuu": "US", "estados unidos": "US",
        "ru": "RU", "russia": "RU", "rusia": "RU", "ng": "NG", "nigeria": "NG"}
SEGMENTO = {"persona": "persona", "individuo": "persona",
            "comercio_pequeno": "comercio_pequeno", "comercio pequeno": "comercio_pequeno", "pyme": "comercio_pequeno",
            "comercio_grande": "comercio_grande", "comercio grande": "comercio_grande", "empresa": "comercio_grande"}
CANAL = {"app": "app", "movil": "app", "web": "web", "pos": "pos", "punto_venta": "pos", "api": "api"}
RESULTADO = {"aprobada": "aprobada", "rechazada": "rechazada", "pendiente": "pendiente"}
TIPO_DISP = {"android": "android", "ios": "ios", "web": "web", "pos": "pos"}
TIPO_LN = {"ip": "ip", "dispositivo": "dispositivo", "device": "dispositivo", "comercio": "comercio",
           "merchant": "comercio", "tarjeta": "tarjeta", "card": "tarjeta"}
COL_LN = {"ip": "ip", "dispositivo": "dispositivo_id", "comercio": "comercio_id", "tarjeta": "tarjeta_enmascarada"}


class ErrorDeDatos(Exception):
    """Los datos no cumplen una validación: el pipeline debe detenerse."""


# ================================================================== 2. LIMPIEZA: reglas escritas
# Cada problema encontrado tiene UNA regla. 'decision' ∈ {corregir, descartar, marcar}.
REGLAS = [
    ("R01", "transacciones", "Misma transaccion_id repetida (copia exacta de la fila)", "descartar",
     "Es la misma fila dos veces; inflaría el entrenamiento y duplicaría el monto en riesgo."),
    ("R02", "usuarios/dispositivos", "Misma clave repetida (usuario_id, dispositivo_id)", "descartar",
     "Copia exacta: se deja una. Si las copias difieren, se conserva la primera y se informa el conflicto."),
    ("R03", "todas", "Categorías escritas de distintas formas (CL, cl, chile, 'Chile '; movil/app; individuo/persona; si/1/TRUE)", "corregir",
     "Mismo valor con distinta escritura. Se normaliza con tablas de equivalencia explícitas; un valor desconocido DETIENE el pipeline."),
    ("R04", "transacciones/dispositivos", "timestamp en 4 formatos; unos con zona (Z, -04:00) y otros sin ella", "corregir",
     "Todo pasa a UTC. Las fechas sin zona y las dd/mm/yyyy son hora de Chile (UTC-4), comprobado con los datos; leídas como UTC correrían la hora 4 h."),
    ("R05", "usuarios/lista_negra", "Fechas de calendario en 3 formatos", "corregir", "Se parsean con los tres formatos conocidos (día primero)."),
    ("R06", "usuarios", "fecha_registro en el futuro", "marcar",
     "Valor imposible; la fecha se anula (nulo) y la edad de cuenta queda nula. La fila sirve: el nulo lo imputa el preprocesador."),
    ("R07", "transacciones", "monto con '$', coma decimal, punto de miles; moneda vacía", "corregir",
     "En CLP el punto es separador de miles ($1.265 = 1265) y en USD es decimal; con eso y el rango (CLP ≥ 901, USD ≤ 313) se infiere la moneda vacía. Todo se lleva a USD."),
    ("R08", "transacciones", "Monto negativo", "corregir",
     "Se toma el valor absoluto (no hay forma de saber si es un error de signo o una reversa) y se marca flag_monto_negativo."),
    ("R09", "transacciones", "Monto igual a cero", "marcar", "Puede ser una verificación de tarjeta; no se descarta, se marca flag_monto_cero."),
    ("R10", "transacciones", "Coordenadas (0, 0) o fuera de rango (|lat|>90, |lon|>180)", "marcar",
     "(0,0) es el Golfo de Guinea, un centinela de dato faltante: se anulan y se marca flag_coord_nula."),
    ("R11", "transacciones", "IP ausente o inválida ('-', octeto > 255, IP truncada)", "marcar",
     "Se anula y se marca flag_ip_invalida; que falte puede ser en sí una señal."),
    ("R12", "usuarios", "nivel_kyc fuera de rango (-1)", "marcar", "Valor imposible: pasa a nulo (el rango válido es 0-3)."),
    ("R13", "usuarios", "edad_cuenta_dias del CSV: foto a la fecha de corte, y negativa en algunos casos", "descartar",
     "No existe al momento de decidir (incluye el futuro). La columna no se usa; la edad se recalcula como fecha de la transacción − fecha_registro; si sale negativa es nula."),
    ("R14", "transacciones", "Duplicado del sistema: el mismo pago APROBADO dos veces (mismo usuario, comercio, dispositivo y monto) a ≤180 s", "descartar",
     "Un cliente no repite un pago que ya pasó. Se descarta el segundo."),
    ("R15", "transacciones", "Reintento del cliente: el mismo pago RECHAZADO/PENDIENTE repetido a ≤180 s", "marcar",
     "Es un intento real y es justo la señal de velocidad que busca el caso. Se conserva y se marca flag_reintento."),
    ("R16", "transacciones", "es_fraude vacío (el 96% de las filas: nadie las revisó)", "marcar",
     "Vacío NO es 'legítima'. Se guarda como sin etiqueta (etiquetada=0). Para métricas se tratan como legítimas y el efecto se declara en el README."),
    ("R17", "transacciones", "revisado_por_analista y motivo_bloqueo se llenan DESPUÉS del veredicto", "marcar",
     "Fuga de información: se conservan en la tabla (ciclo de reetiquetado) pero NO entran al modelo."),
    ("R18", "lista_negra", "Valores con espacios sobrantes; entradas dadas de baja (vigente=0)", "corregir",
     "strip() al valor. Las entradas dadas de baja no participan del cruce. Solo cuenta una entrada si ya existía el día de la transacción."),
    ("R19", "variables del modelo", "Nulos en columnas que alimentan el modelo", "marcar",
     "NO se rellenan aquí: se imputan en entrenar.py con medianas calculadas solo con entrenamiento (evita fuga) y con un indicador 'faltaba'. Así la imputación viaja con el modelo."),
]


def clave(x):
    """minúsculas, sin tildes, sin espacios sobrantes: 'Perú ' -> 'peru'"""
    if pd.isna(x):
        return x
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def mapear(s, mapa, nombre=""):
    k = s.map(clave)
    out = k.map(mapa)
    sin_mapa = s[k.notna() & out.isna()].unique()
    if len(sin_mapa):
        raise ErrorDeDatos(f"{nombre}: valores sin equivalencia {list(sin_mapa)[:10]}; agrégalos a la tabla de equivalencias")
    return out


def a_bool(s, nombre=""):
    k = s.map(clave)
    out = k.map(BOOL)
    sin_mapa = s[k.notna() & out.isna()].unique()
    if len(sin_mapa):
        raise ErrorDeDatos(f"{nombre}: valores booleanos desconocidos {list(sin_mapa)[:10]}")
    return out.astype(float)


def parse_fecha(s):
    s = s.astype("string")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for patron, fmt in [(r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"), (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"), (r"^\d{2}-\d{2}-\d{4}$", "%d-%m-%Y")]:
        m = s.str.match(patron, na=False)
        if m.any():
            out[m] = pd.to_datetime(s[m], format=fmt, errors="coerce")
    return out


def parse_ts_utc(s):
    """Instante en UTC (sin zona adjunta). Sin zona y dd/mm/yyyy = hora de Chile (UTC-4)."""
    s = s.astype("string")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    con_zona = s.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False)
    barra = s.str.match(r"^\d{2}/\d{2}/\d{4}", na=False)
    naive = ~con_zona & ~barra & s.notna()
    desfase = pd.Timedelta(hours=OFFSET_CHILE_H)
    if con_zona.any():
        out[con_zona] = pd.to_datetime(s[con_zona], utc=True, format="ISO8601", errors="coerce").dt.tz_localize(None)
    if barra.any():
        out[barra] = pd.to_datetime(s[barra], format="%d/%m/%Y %H:%M", errors="coerce") + desfase
    if naive.any():
        out[naive] = pd.to_datetime(s[naive], format="%Y-%m-%d %H:%M:%S", errors="coerce") + desfase
    return out


def parse_monto(m):
    """3 dígitos tras el separador ($1.265) -> punto de MILES (patrón CLP); 2 dígitos (47.33, 6,36) -> decimal (patrón USD)."""
    s = m.astype("string").str.strip().str.replace("$", "", regex=False)
    miles = s.str.match(r"^-?\d{1,3}(?:\.\d{3})+$", na=False)
    s2 = s.mask(miles, s.str.replace(".", "", regex=False)).str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce"), miles


def ip_valida(ip):
    if pd.isna(ip):
        return False
    p = str(ip).split(".")
    return len(p) == 4 and all(o.isdigit() and 0 <= int(o) <= 255 for o in p)


# ------------------------------------------------------------------ normalización fila a fila (compartida con el servicio)
def normalizar_transacciones(tx):
    """R03-R11 sobre transacciones crudas. Sirve igual para un lote o para UNA transacción que llega a la API."""
    tx = tx.reset_index(drop=True)
    o = pd.DataFrame({"transaccion_id": tx["transaccion_id"], "usuario_id": tx["usuario_id"],
                      "comercio_id": tx["comercio_id"], "dispositivo_id": tx["dispositivo_id"]})
    o["ts_utc"] = parse_ts_utc(tx["timestamp"])
    valor, miles = parse_monto(tx["monto"])
    moneda = tx["moneda"].map(lambda v: str(v).strip().upper() if pd.notna(v) else np.nan)
    inferida = moneda.isna()
    moneda = moneda.mask(inferida & (miles | (valor.abs() >= UMBRAL_MONEDA)), "CLP")
    moneda = moneda.mask(moneda.isna(), "USD")
    o["monto_valor"] = valor
    o["moneda_final"] = moneda
    o["moneda_inferida"] = inferida
    o["monto_negativo"] = valor < 0
    o["monto_cero"] = valor == 0
    o["monto_usd"] = np.where(moneda == "CLP", valor.abs() / TASA_CLP_USD, valor.abs())
    o["canal"] = mapear(tx["canal"], CANAL, "canal")
    o["pais_tx"] = mapear(tx["pais_transaccion"], PAIS, "pais_transaccion")
    o["mcc"] = tx["mcc"].astype(int).astype(str)
    # El resultado de la autorización NO existe al decidir (la API no lo recibe); solo sirve para limpiar el histórico
    o["resultado"] = mapear(tx["resultado_autorizacion"], RESULTADO, "resultado_autorizacion") if "resultado_autorizacion" in tx.columns else np.nan
    lat, lon = pd.to_numeric(tx["latitud"], errors="coerce"), pd.to_numeric(tx["longitud"], errors="coerce")
    coord_mala = ((lat == 0) & (lon == 0)) | (lat.abs() > 90) | (lon.abs() > 180) | lat.isna() | lon.isna()
    o["coord_nula"] = coord_mala
    o["latitud"] = lat.mask(coord_mala)
    o["longitud"] = lon.mask(coord_mala)
    ok_ip = tx["ip"].map(ip_valida)
    o["ip"] = tx["ip"].where(ok_ip)
    o["ip_ausente_o_invalida"] = (~ok_ip).astype(float)
    o["tarjeta_enmascarada"] = tx["tarjeta_enmascarada"]      # solo para la lista negra; el modelo NUNCA la ve (PCI)
    o["intentos_previos_1h"] = pd.to_numeric(tx["intentos_previos_1h"], errors="coerce")
    return o


def normalizar_usuarios(u):
    o = pd.DataFrame({"usuario_id": u["usuario_id"].values})
    o["pais_usuario"] = mapear(u["pais"].reset_index(drop=True), PAIS, "usuarios.pais").values
    o["segmento"] = mapear(u["segmento"].reset_index(drop=True), SEGMENTO, "usuarios.segmento").values
    o["verificado"] = a_bool(u["verificado"].reset_index(drop=True), "usuarios.verificado").values
    o["telefono_verificado"] = u["telefono_verificado"].astype(float).values
    kyc = u["nivel_kyc"].astype(float).reset_index(drop=True)
    o["nivel_kyc"] = kyc.where(kyc.between(0, 3)).values                                    # R12
    o["email_dominio"] = u["email_dominio"].values
    o["dominio_desechable"] = np.where(u["email_dominio"].isna(), np.nan, u["email_dominio"].isin(DESECHABLES).astype(float))
    reg = parse_fecha(u["fecha_registro"].reset_index(drop=True))
    o["fecha_registro_original"] = reg.values
    o["fecha_registro"] = reg.mask(reg > pd.Timestamp.now()).values                          # R06
    o["snap_transacciones_previas"] = u["transacciones_previas"].astype(float).values        # foto a la fecha de corte: NO es variable
    return o


def normalizar_dispositivos(d):
    o = pd.DataFrame({"dispositivo_id": d["dispositivo_id"].values})
    o["tipo_dispositivo"] = mapear(d["tipo"].reset_index(drop=True), TIPO_DISP, "dispositivos.tipo").values
    o["pais_disp"] = mapear(d["pais_declarado"].reset_index(drop=True), PAIS, "dispositivos.pais_declarado").values
    o["emulador"] = d["emulador"].astype(float).values
    o["sistema_operativo"] = d["sistema_operativo"].values
    o["primera_vez_utc"] = parse_ts_utc(d["primera_vez"].reset_index(drop=True)).values
    o["snap_usuarios_distintos"] = d["usuarios_distintos"].astype(float).values              # foto a la fecha de corte: NO es variable
    return o


def normalizar_lista_negra(ln):
    o = pd.DataFrame({"tipo": mapear(ln["tipo"], TIPO_LN, "lista_negra.tipo").values, "valor": ln["valor"].str.strip().values,
                      "fecha_alta": parse_fecha(ln["fecha_alta"]).values, "motivo": ln["motivo"].values, "vigente": ln["vigente"].values})
    o["activa"] = o["vigente"] != 0                                                          # nulo = se asume activa; 0 = dada de baja
    o["dura"] = (o["vigente"] == 1) & (o["motivo"] == "solicitud_cumplimiento")              # SUPUESTO: solo los pedidos de cumplimiento son regla normativa
    return o


# ================================================================== 4. VARIABLES DEL MODELO
# (nombre, tipo, columnas de origen, cómo se calcula)
VARIABLES = [
    ("log_monto_usd", "num", "transacciones.monto, moneda", "log(1 + monto en USD). CLP se divide por TASA_CLP_USD; la moneda vacía se infiere (R07)"),
    ("hora_local", "num", "transacciones.timestamp", "hora (0-23) en hora de Chile = UTC-4, tras llevar el timestamp a UTC (R04)"),
    ("es_madrugada", "num", "transacciones.timestamp", "1 si hora_local < 6 (el 49% del fraude ocurre 00-06)"),
    ("intentos_previos_1h", "num", "transacciones.intentos_previos_1h", "tal cual: lo calcula el core sobre TODO el tráfico (recalcularlo con la muestra estratificada sería engañoso)"),
    ("nivel_kyc", "num", "usuarios.nivel_kyc", "0-3; el -1 pasa a nulo (R12)"),
    ("verificado", "num", "usuarios.verificado", "0/1 tras normalizar si/no/TRUE (R03)"),
    ("telefono_verificado", "num", "usuarios.telefono_verificado", "0/1"),
    ("dominio_desechable", "num", "usuarios.email_dominio", "1 si el dominio está en la lista de correos desechables; nulo si no hay correo"),
    ("edad_cuenta_dias", "num", "usuarios.fecha_registro, transacciones.timestamp", "(fecha de la transacción − fecha_registro) en días; negativa o sin registro = nulo (R06, R13)"),
    ("emulador", "num", "dispositivos.emulador", "0/1 del tag del cliente"),
    ("pais_distinto_usuario", "num", "transacciones.pais_transaccion, usuarios.pais", "1 si el país de la transacción difiere del país del usuario; nulo si el usuario no tiene perfil"),
    ("pais_distinto_dispositivo", "num", "transacciones.pais_transaccion, dispositivos.pais_declarado", "1 si difiere del país declarado del dispositivo; nulo si no hay dispositivo"),
    ("ip_ausente_o_invalida", "num", "transacciones.ip", "1 si la IP falta o es inválida (R11)"),
    ("dispositivo_ausente", "num", "transacciones.dispositivo_id", "1 si la transacción no trae dispositivo"),
    ("lista_negra_interna", "num", "lista_negra + transacciones.ip/dispositivo_id/comercio_id", "1 si hay coincidencia con una entrada activa NO dura que ya existía el día de la transacción (R18)"),
    ("moneda_final", "cat", "transacciones.moneda, monto", "CLP / USD (R07)"),
    ("canal", "cat", "transacciones.canal", "app / web / pos / api (R03)"),
    ("pais_tx", "cat", "transacciones.pais_transaccion", "código ISO de 2 letras (R03)"),
    ("segmento", "cat", "usuarios.segmento", "persona / comercio_pequeno / comercio_grande (R03)"),
    ("tipo_dispositivo", "cat", "dispositivos.tipo", "android / ios / web / pos (R03)"),
    ("mcc", "cat", "transacciones.mcc", "código de rubro como categoría"),
]
FEATURES_NUM = [v[0] for v in VARIABLES if v[1] == "num"]
FEATURES_CAT = [v[0] for v in VARIABLES if v[1] == "cat"]
FEATURES = FEATURES_NUM + FEATURES_CAT
EXCLUIDAS = [
    ("es_fraude", "es la etiqueta, no una variable"),
    ("revisado_por_analista, motivo_bloqueo", "se llenan después del veredicto (fuga, R17)"),
    ("resultado_autorizacion", "no existe antes de autorizar"),
    ("usuarios.edad_cuenta_dias, usuarios.transacciones_previas, dispositivos.usuarios_distintos", "fotos a la fecha de corte: incluyen el futuro (R13)"),
    ("transacciones.tarjeta_enmascarada", "identificador, no característica (PCI-DSS); solo la regla de lista negra la consulta"),
    ("usuario_id, transaccion_id, comercio_id, dispositivo_id", "identificadores: el modelo memorizaría entidades en vez de patrones"),
    ("coordenadas, ip", "ruido para esta versión (ver README); la IP solo entra como 'ausente o inválida'"),
]


def aplicar_lista_negra(tx, ln):
    """Coincidencia puntual en el tiempo: cuenta solo si la entrada ya estaba dada de alta el día de la transacción."""
    base = pd.DataFrame({"_i": np.arange(len(tx)), "dia": tx["ts_utc"].dt.normalize().values})
    dura = np.zeros(len(tx), dtype=bool)
    interna = np.zeros(len(tx), dtype=bool)
    for tipo, col in COL_LN.items():
        e = ln[(ln["tipo"] == tipo) & ln["activa"]][["valor", "fecha_alta", "dura"]]
        m = base.assign(valor=tx[col].values).merge(e, on="valor", how="inner")
        m = m[m["fecha_alta"] <= m["dia"]]
        dura[m.loc[m["dura"], "_i"].unique()] = True
        interna[m.loc[~m["dura"], "_i"].unique()] = True
    return dura, interna


def construir_variables(tx_norm, usr, dsp, ln):
    """Une las tablas por sus claves (left join: nunca se pierde una transacción) y calcula las variables."""
    cols_usr = ["usuario_id", "pais_usuario", "segmento", "verificado", "telefono_verificado", "nivel_kyc", "dominio_desechable", "fecha_registro"]
    x = tx_norm.merge(usr[cols_usr], on="usuario_id", how="left")      # solo lo que necesita: sirve con la tabla tratada que carga el servicio
    x = x.merge(dsp[["dispositivo_id", "tipo_dispositivo", "pais_disp", "emulador"]], on="dispositivo_id", how="left")
    x["hora_local"] = (x["ts_utc"] - pd.Timedelta(hours=OFFSET_CHILE_H)).dt.hour.astype(float)
    x["es_madrugada"] = (x["hora_local"] < 6).astype(float)
    x["en_turno"] = (x["hora_local"] >= TURNO[0]) & (x["hora_local"] < TURNO[1])
    x["log_monto_usd"] = np.log1p(x["monto_usd"])
    edad = (x["ts_utc"] - x["fecha_registro"]).dt.days.astype(float)
    x["edad_cuenta_dias"] = edad.where(edad >= 0)
    x["pais_distinto_usuario"] = np.where(x["pais_usuario"].isna(), np.nan, (x["pais_tx"] != x["pais_usuario"]).astype(float))
    x["pais_distinto_dispositivo"] = np.where(x["pais_disp"].isna(), np.nan, (x["pais_tx"] != x["pais_disp"]).astype(float))
    x["dispositivo_ausente"] = x["dispositivo_id"].isna().astype(float)
    dura, interna = aplicar_lista_negra(x, ln)
    x["lista_negra_dura"] = dura
    x["lista_negra_interna"] = interna.astype(float)
    for c in FEATURES_CAT:
        x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
    return x


# ================================================================== 1. PERFILADO
def perfilar(df, clave_id=None):
    return {"filas": int(len(df)),
            "duplicados_exactos": int(df.duplicated().sum()),
            "claves_duplicadas": int(df[clave_id].duplicated().sum()) if clave_id and clave_id in df.columns else None,
            "celdas_nulas": int(df.isna().sum().sum()),
            "columnas": {c: {"tipo": str(df[c].dtype), "nulos": int(df[c].isna().sum()), "pct_nulos": round(float(df[c].isna().mean() * 100), 2),
                             "unicos": int(df[c].nunique(dropna=True))} for c in df.columns},
            "n_columnas": int(df.shape[1])}


# ================================================================== tratamiento por archivo (lote)
def tratar_transacciones(raw, cuentas):
    n0 = len(raw)
    tx = raw.drop_duplicates(subset="transaccion_id").reset_index(drop=True)                      # R01
    cuentas["R01"] = n0 - len(tx)
    n = normalizar_transacciones(tx)
    cuentas["R03"] = int((tx["canal"] != n["canal"]).sum() + (tx["pais_transaccion"] != n["pais_tx"]).sum())   # valores reescritos
    cuentas["R04"] = int((~tx["timestamp"].str.contains(r"(?:Z|[+-]\d{2}:\d{2})$")).sum())      # filas sin zona llevadas a UTC
    cuentas["R07"] = int(n["moneda_inferida"].sum())
    cuentas["R08"] = int(n["monto_negativo"].sum())
    cuentas["R09"] = int(n["monto_cero"].sum())
    cuentas["R10"] = int(n["coord_nula"].sum())
    cuentas["R11"] = int(n["ip_ausente_o_invalida"].sum())
    n["es_fraude"] = a_bool(tx["es_fraude"], "es_fraude")
    n["revisado_por_analista"] = a_bool(tx["revisado_por_analista"], "revisado_por_analista")
    n["motivo_bloqueo"] = tx["motivo_bloqueo"]
    # R14/R15: casi-duplicados
    orden = n.sort_values(["usuario_id", "comercio_id", "dispositivo_id", "monto_valor", "resultado", "ts_utc"]).index
    g = n.loc[orden].groupby(["usuario_id", "comercio_id", "dispositivo_id", "monto_valor", "resultado"], dropna=False)
    repetida = (g["ts_utc"].diff().dt.total_seconds() <= VENTANA_DUPLICADO_S).reindex(n.index).fillna(False).astype(bool)
    n["flag_reintento"] = repetida & (n["resultado"] != "aprobada")
    dup_sistema = repetida & (n["resultado"] == "aprobada")
    cuentas["R14"] = int(dup_sistema.sum())
    cuentas["R15"] = int(n["flag_reintento"].sum())
    n = n[~dup_sistema].reset_index(drop=True)
    n["etiquetada"] = n["es_fraude"].notna()                                                     # R16
    n["y"] = n["es_fraude"].fillna(0).astype(int)
    cuentas["R16"] = int((~n["etiquetada"]).sum())
    cuentas["R17"] = int(n["motivo_bloqueo"].notna().sum())
    return n


def tratar_usuarios(raw, cuentas):
    u = raw.drop_duplicates(subset="usuario_id", keep="first")
    exactos = int(raw.duplicated().sum())
    cuentas["R02_usuarios"] = len(raw) - len(u)
    cuentas["R02_usuarios_conflictivos"] = int(raw.duplicated(subset="usuario_id").sum()) - exactos
    n = normalizar_usuarios(u)
    cuentas["R06"] = int((n["fecha_registro_original"] > pd.Timestamp.now()).sum())
    kyc = u["nivel_kyc"].astype(float)
    cuentas["R12"] = int((kyc.notna() & ~kyc.between(0, 3)).sum())
    cuentas["R13"] = int((u["edad_cuenta_dias"] < 0).sum())
    return n


def tratar_dispositivos(raw, cuentas):
    d = raw.drop_duplicates(subset="dispositivo_id", keep="first")
    cuentas["R02_dispositivos"] = len(raw) - len(d)
    cuentas["R02_dispositivos_conflictivos"] = int(raw.duplicated(subset="dispositivo_id").sum()) - int(raw.duplicated().sum())
    return normalizar_dispositivos(d)


def tratar_lista_negra(raw, cuentas):
    cuentas["R18_espacios"] = int((raw["valor"] != raw["valor"].str.strip()).sum())
    n = normalizar_lista_negra(raw)
    cuentas["R18_dadas_de_baja"] = int((~n["activa"]).sum())
    return n


# ================================================================== 3. INTEGRACIÓN
def informe_integracion(tx, usr, dsp, ln):
    inf = {}
    sin_usr = ~tx["usuario_id"].isin(usr["usuario_id"])
    inf["transacciones_sin_usuario"] = {"filas": int(sin_usr.sum()), "usuarios_distintos": int(tx.loc[sin_usr, "usuario_id"].nunique()),
        "decision": "se CONSERVAN (left join): en producción llegarán usuarios sin perfil. Las variables del perfil quedan nulas y el preprocesador las imputa."}
    dnull = tx["dispositivo_id"].isna()
    dno = tx["dispositivo_id"].notna() & ~tx["dispositivo_id"].isin(dsp["dispositivo_id"])
    inf["transacciones_sin_dispositivo_id"] = {"filas": int(dnull.sum()), "decision": "se CONSERVAN; dispositivo_ausente=1 (es una señal)."}
    inf["transacciones_con_dispositivo_no_registrado"] = {"filas": int(dno.sum()), "decision": "se CONSERVAN; las variables del dispositivo quedan nulas."}
    inf["usuarios_sin_transacciones"] = {"filas": int((~usr["usuario_id"].isin(tx["usuario_id"])).sum()),
        "decision": "se conservan en la tabla de usuarios (el servicio de carga los necesita) pero no generan filas del modelo."}
    inf["dispositivos_sin_transacciones"] = {"filas": int((~dsp["dispositivo_id"].isin(tx["dispositivo_id"])).sum()),
        "decision": "se conservan en la tabla de dispositivos; no generan filas del modelo."}
    por_tipo = {}
    for tipo, col in COL_LN.items():
        e = ln[ln["tipo"] == tipo]
        por_tipo[tipo] = {"entradas": int(len(e)), "sin_ninguna_transaccion": int((~e["valor"].isin(tx[col])).sum())}
    inutiles = [t for t, v in por_tipo.items() if v["entradas"] > 0 and v["sin_ninguna_transaccion"] == v["entradas"]]
    inf["lista_negra_entradas_sin_pareja"] = {"por_tipo": por_tipo,
        "decision": "se conservan (la lista es una tabla de consulta, no se espera que todo valor aparezca en esta muestra)."
                    + (f" ATENCIÓN: ninguna entrada de tipo {', '.join(inutiles)} coincide con alguna transacción de la muestra; "
                       "esa parte de la lista no puede aportar al modelo ni a la regla dura aquí." if inutiles else "")}
    return inf


# ================================================================== 5. SEPARACIÓN
def separar_por_fecha(x):
    x = x.sort_values(["ts_utc", "transaccion_id"]).reset_index(drop=True)
    n = len(x)
    i1, i2 = int(n * FRACCIONES_SPLIT[0]), int(n * (FRACCIONES_SPLIT[0] + FRACCIONES_SPLIT[1]))
    x["split"] = np.where(np.arange(n) < i1, "train", np.where(np.arange(n) < i2, "val", "test"))
    return x


# ================================================================== utilidades de salida
def tabla_md(filas, columnas):
    out = ["| " + " | ".join(columnas) + " |", "|" + "|".join(["---"] * len(columnas)) + "|"]
    for f in filas:
        out.append("| " + " | ".join(str(f[c]).replace("|", "/").replace("\n", " ") for c in columnas) + " |")
    return "\n".join(out)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def escribir_reporte(salida, antes, despues, cuentas, integ, x, splits, manifiesto):
    L = ["# Reporte de tratamiento de datos", f"_Generado por `datos/preparar.py` el {manifiesto['generado_en']}_", ""]
    L += ["## 1. Perfilado antes y después", ""]
    filas = []
    for k in ARCHIVOS:
        a, d = antes[k], despues[k]
        filas.append({"archivo": k, "filas antes → después": f"{a['filas']:,} → {d['filas']:,}", "columnas antes → después": f"{a['n_columnas']} → {d['n_columnas']}",
                      "duplicados exactos": f"{a['duplicados_exactos']} → {d['duplicados_exactos']}",
                      "claves duplicadas": f"{a['claves_duplicadas']} → {d['claves_duplicadas']}" if a["claves_duplicadas"] is not None else "—",
                      "celdas nulas": f"{a['celdas_nulas']:,} → {d['celdas_nulas']:,}"})
    L += [tabla_md(filas, list(filas[0])), ""]
    for k in ARCHIVOS:
        L += [f"### {k}: tipos y nulos por columna", ""]
        for etiqueta, perfil in [("Antes", antes[k]), ("Después", despues[k])]:
            fs = [{"columna": c, "tipo": v["tipo"], "nulos": v["nulos"], "% nulos": v["pct_nulos"], "únicos": v["unicos"]} for c, v in perfil["columnas"].items()]
            L += [f"**{etiqueta}**", "", tabla_md(fs, ["columna", "tipo", "nulos", "% nulos", "únicos"]), ""]
    L += ["## 2. Limpieza: una regla por problema", ""]
    claves_cuenta = {"R01": ["R01"], "R02": ["R02_usuarios", "R02_dispositivos"], "R03": ["R03"], "R04": ["R04"], "R06": ["R06"], "R07": ["R07"], "R08": ["R08"],
                     "R09": ["R09"], "R10": ["R10"], "R11": ["R11"], "R12": ["R12"], "R13": ["R13"], "R14": ["R14"], "R15": ["R15"], "R16": ["R16"],
                     "R17": ["R17"], "R18": ["R18_espacios", "R18_dadas_de_baja"]}
    fs = []
    for rid, donde, problema, decision, porque in REGLAS:
        afect = " + ".join(str(cuentas[c]) for c in claves_cuenta.get(rid, []) if c in cuentas) or "—"
        fs.append({"regla": rid, "dónde": donde, "problema": problema, "decisión": decision, "por qué": porque, "filas afectadas": afect})
    L += [tabla_md(fs, ["regla", "dónde", "problema", "decisión", "por qué", "filas afectadas"]), ""]
    L += [f"Usuarios/dispositivos con copias que difieren entre sí (conflicto): {cuentas.get('R02_usuarios_conflictivos', 0)} / {cuentas.get('R02_dispositivos_conflictivos', 0)}.", ""]
    L += ["## 3. Integración (claves y registros sin pareja)", "",
          "Claves: `transacciones.usuario_id → usuarios.usuario_id`, `transacciones.dispositivo_id → dispositivos.dispositivo_id`; la `lista_negra` se cruza por valor "
          "(ip, dispositivo, comercio, tarjeta) y solo si la entrada ya existía el día de la transacción.", ""]
    fs = [{"caso": k, "filas": v.get("filas", "—"), "qué se hace": v["decision"]} for k, v in integ.items() if k != "lista_negra_entradas_sin_pareja"]
    L += [tabla_md(fs, ["caso", "filas", "qué se hace"]), "", "Lista negra (entradas sin ninguna transacción en la muestra):", ""]
    fs = [{"tipo": t, "entradas": v["entradas"], "sin transacción": v["sin_ninguna_transaccion"]} for t, v in integ["lista_negra_entradas_sin_pareja"]["por_tipo"].items()]
    L += [tabla_md(fs, ["tipo", "entradas", "sin transacción"]), "", integ["lista_negra_entradas_sin_pareja"]["decision"], ""]
    L += ["## 4. Variables de entrada del modelo", ""]
    L += [tabla_md([{"variable": n, "tipo": t, "sale de": f, "cómo se calcula": c} for n, t, f, c in VARIABLES], ["variable", "tipo", "sale de", "cómo se calcula"]), "",
          "**Lo que NO entra al modelo:**", "", tabla_md([{"columna": c, "motivo": m} for c, m in EXCLUIDAS], ["columna", "motivo"]), ""]
    L += ["## 5. Separación entrenamiento / validación / prueba (por fecha, antes de entrenar)", "",
          "Se ordena por `ts_utc` y se corta en 60% / 20% / 20%. Un reparto al azar mezclaría semanas y daría una nota inflada: en producción el modelo siempre predice el futuro con datos del pasado.", ""]
    L += [tabla_md(splits, ["conjunto", "filas", "desde", "hasta", "fraudes", "% fraude", "etiquetadas"]), ""]
    L += ["## 6. Salidas", "", tabla_md([{"archivo": k, "sha256 (16)": v} for k, v in manifiesto["salidas"].items()], ["archivo", "sha256 (16)"]), ""]
    (salida / "reporte_tratamiento.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tratamiento de datos de Kipu Pagos")
    ap.add_argument("--entrada", type=Path, default=RAIZ / "datos" / "crudo")
    ap.add_argument("--salida", type=Path, default=RAIZ / "datos" / "tratado")
    args = ap.parse_args(argv)
    try:
        crudo = {}
        for k in ARCHIVOS:
            ruta = args.entrada / f"{k}.csv"
            if not ruta.exists():
                raise ErrorDeDatos(f"Falta el archivo {ruta}")
            crudo[k] = pd.read_csv(ruta)
        antes = {k: perfilar(df, ARCHIVOS[k]) for k, df in crudo.items()}

        # ---- verificación de supuestos que sostienen la limpieza (si los datos cambian, el pipeline avisa)
        v, miles = parse_monto(crudo["transacciones"]["monto"])
        mon = crudo["transacciones"]["moneda"].str.strip().str.upper()
        usd_max, clp_min = v[mon == "USD"].abs().max(), v[(mon == "CLP") & (v.abs() > 0)].abs().min()
        if not (usd_max < UMBRAL_MONEDA <= clp_min):
            raise ErrorDeDatos(f"El supuesto de moneda ya no se cumple (USD máx {usd_max}, CLP mín {clp_min}, umbral {UMBRAL_MONEDA})")

        cuentas = {}
        tx = tratar_transacciones(crudo["transacciones"], cuentas)
        usr = tratar_usuarios(crudo["usuarios"], cuentas)
        dsp = tratar_dispositivos(crudo["dispositivos"], cuentas)
        ln = tratar_lista_negra(crudo["lista_negra"], cuentas)
        integ = informe_integracion(tx, usr, dsp, ln)

        x = construir_variables(tx, usr, dsp, ln)
        faltan = [c for c in FEATURES if c not in x.columns]
        if faltan:
            raise ErrorDeDatos(f"No se pudieron calcular las variables {faltan}")
        x["flag_registro_posterior_a_tx"] = (x["ts_utc"] < x["fecha_registro"]).astype(int)
        cuentas["R13_recalculada_negativa"] = int(x["flag_registro_posterior_a_tx"].sum())
        x = separar_por_fecha(x)

        splits = []
        for nombre, etiqueta in [("train", "entrenamiento"), ("val", "validación"), ("test", "prueba")]:
            d = x[x["split"] == nombre]
            if int(d["y"].sum()) < MIN_FRAUDES_POR_CONJUNTO:
                raise ErrorDeDatos(f"El conjunto de {etiqueta} tiene {int(d['y'].sum())} fraudes (mínimo {MIN_FRAUDES_POR_CONJUNTO})")
            splits.append({"conjunto": etiqueta, "filas": f"{len(d):,}", "desde": f"{d['ts_utc'].min():%Y-%m-%d}", "hasta": f"{d['ts_utc'].max():%Y-%m-%d}",
                           "fraudes": int(d["y"].sum()), "% fraude": f"{d['y'].mean():.2%}", "etiquetadas": int(d["etiquetada"].sum())})

        # ---- salidas: tablas tratadas para el servicio de carga
        args.salida.mkdir(parents=True, exist_ok=True)
        t_out = tx.rename(columns={"moneda_final": "moneda", "monto_negativo": "flag_monto_negativo", "monto_cero": "flag_monto_cero",
                                   "moneda_inferida": "flag_moneda_inferida", "coord_nula": "flag_coord_nula"})
        t_out["flag_ip_invalida"] = t_out["ip_ausente_o_invalida"].astype(int)
        t_out = t_out[["transaccion_id", "usuario_id", "comercio_id", "dispositivo_id", "tarjeta_enmascarada", "ts_utc", "monto_valor", "moneda", "monto_usd",
                       "canal", "pais_tx", "mcc", "resultado", "latitud", "longitud", "ip", "intentos_previos_1h", "revisado_por_analista", "motivo_bloqueo",
                       "es_fraude", "etiquetada", "flag_moneda_inferida", "flag_monto_negativo", "flag_monto_cero", "flag_coord_nula", "flag_ip_invalida", "flag_reintento"]]
        u_out = usr.drop(columns=["fecha_registro_original"]).assign(flag_fecha_registro_futura=usr["fecha_registro_original"] > pd.Timestamp.now())
        tablas = {"transacciones": t_out, "usuarios": u_out, "dispositivos": dsp, "lista_negra": ln}
        for nombre, df in tablas.items():
            df.to_csv(args.salida / f"{nombre}.csv", index=False)
        cols_modelo = ["transaccion_id", "ts_utc", "split", "y", "etiquetada", "monto_usd", "hora_local", "en_turno", "lista_negra_dura", "flag_registro_posterior_a_tx"] + FEATURES
        x[cols_modelo].to_csv(args.salida / "variables_modelo.csv", index=False)
        x.loc[x["split"] == "test", ["transaccion_id"]].to_csv(args.salida / "test_fijo_ids.csv", index=False)
        (args.salida / "variables.json").write_text(json.dumps({
            "numericas": FEATURES_NUM, "categoricas": FEATURES_CAT,
            "especificacion": [{"variable": n, "tipo": t, "sale_de": f, "calculo": c} for n, t, f, c in VARIABLES],
            "excluidas": [{"columna": c, "motivo": m} for c, m in EXCLUIDAS],
            "parametros": {"offset_chile_h": OFFSET_CHILE_H, "turno": list(TURNO), "tasa_clp_usd": TASA_CLP_USD}}, ensure_ascii=False, indent=2))

        despues = {k: perfilar(tablas[k], ARCHIVOS[k]) for k in ARCHIVOS}
        manifiesto = {"generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "entradas": {k: sha(args.entrada / f"{k}.csv") for k in ARCHIVOS}, "salidas": {}}
        for f in ["transacciones", "usuarios", "dispositivos", "lista_negra", "variables_modelo"]:
            manifiesto["salidas"][f"{f}.csv"] = sha(args.salida / f"{f}.csv")
        escribir_reporte(args.salida, antes, despues, cuentas, integ, x, splits, manifiesto)
        (args.salida / "reporte_tratamiento.json").write_text(json.dumps(
            {"perfil_antes": antes, "perfil_despues": despues, "reglas": [dict(zip(["id", "dónde", "problema", "decision", "por_que"], r)) for r in REGLAS],
             "filas_afectadas": cuentas, "integracion": integ, "separacion": splits}, ensure_ascii=False, indent=2, default=str))
        (args.salida / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2))
    except ErrorDeDatos as e:
        print(f"ERROR DE DATOS: {e}", file=sys.stderr)
        return 2

    print(f"preparar.py OK · {len(tx):,} transacciones tratadas ({antes['transacciones']['filas']:,} de entrada) · "
          f"{int(x['y'].sum())} fraudes · salidas en {args.salida}")
    for s in splits:
        print(f"   {s['conjunto']:<14} {s['filas']:>7} filas · {s['desde']} → {s['hasta']} · {s['fraudes']} fraudes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
