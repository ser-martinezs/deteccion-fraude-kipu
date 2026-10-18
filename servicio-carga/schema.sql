-- Guardián Centinela · esquema de la base (PostgreSQL 16). Idempotente: se puede ejecutar N veces.
-- Las tablas usuarios, dispositivos, lista_negra y transacciones guardan la salida de datos/preparar.py
-- (datos/tratado/*.csv) con los mismos nombres de columna. predicciones la escribe el microservicio de dominio.
SET TIME ZONE 'UTC';

CREATE TABLE IF NOT EXISTS usuarios (
    usuario_id                  TEXT PRIMARY KEY,
    pais_usuario                CHAR(2),
    segmento                    TEXT CHECK (segmento IN ('persona','comercio_pequeno','comercio_grande')),
    verificado                  REAL,
    telefono_verificado         REAL,
    nivel_kyc                   REAL CHECK (nivel_kyc BETWEEN 0 AND 3),
    email_dominio               TEXT,
    dominio_desechable          REAL,
    fecha_registro              DATE,
    snap_transacciones_previas  REAL,
    flag_fecha_registro_futura  BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS dispositivos (
    dispositivo_id           TEXT PRIMARY KEY,
    tipo_dispositivo         TEXT CHECK (tipo_dispositivo IN ('android','ios','web','pos')),
    pais_disp                CHAR(2),
    emulador                 REAL,
    sistema_operativo        TEXT,
    primera_vez_utc          TIMESTAMPTZ,
    snap_usuarios_distintos  REAL
);

-- Clave natural (tipo, valor, fecha_alta): un mismo valor puede vetarse dos veces en fechas distintas
CREATE TABLE IF NOT EXISTS lista_negra (
    tipo        TEXT NOT NULL CHECK (tipo IN ('ip','dispositivo','comercio','tarjeta')),
    valor       TEXT NOT NULL,
    fecha_alta  DATE NOT NULL,
    motivo      TEXT,
    vigente     REAL,
    activa      BOOLEAN NOT NULL,
    dura        BOOLEAN NOT NULL,      -- solicitud de cumplimiento: bloqueo obligatorio (RF4)
    PRIMARY KEY (tipo, valor, fecha_alta)
);
CREATE INDEX IF NOT EXISTS ix_ln_valor ON lista_negra (tipo, valor) WHERE activa;

-- Historial tratado (transacciones evaluadas por el motor actual, con su etiqueta si existe)
CREATE TABLE IF NOT EXISTS transacciones (
    transaccion_id        TEXT PRIMARY KEY,
    usuario_id            TEXT NOT NULL,          -- FK lógica: hay usuarios que no vienen en la muestra
    comercio_id           TEXT,
    dispositivo_id        TEXT,
    tarjeta_enmascarada   TEXT,                   -- solo ****NNNN: el PAN nunca entra (RNF5)
    ts_utc                TIMESTAMPTZ NOT NULL,
    monto_valor           NUMERIC(14,2) NOT NULL,
    moneda                CHAR(3) NOT NULL CHECK (moneda IN ('CLP','USD')),
    monto_usd             NUMERIC(14,4) NOT NULL,
    canal                 TEXT NOT NULL,
    pais_tx               CHAR(2),
    mcc                   TEXT NOT NULL,
    resultado             TEXT,
    latitud               DOUBLE PRECISION,
    longitud              DOUBLE PRECISION,
    ip                    TEXT,
    intentos_previos_1h   REAL,
    revisado_por_analista REAL,                   -- post-decisión: no se usa para entrenar (fuga)
    motivo_bloqueo        TEXT,                   -- post-decisión: no se usa para entrenar (fuga)
    es_fraude             REAL,                   -- NULL = sin revisar (no significa legítima)
    etiquetada            BOOLEAN NOT NULL,
    flag_moneda_inferida  BOOLEAN NOT NULL,
    flag_monto_negativo   BOOLEAN NOT NULL,
    flag_monto_cero       BOOLEAN NOT NULL,
    flag_coord_nula       BOOLEAN NOT NULL,
    flag_ip_invalida      SMALLINT NOT NULL,
    flag_reintento        BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tx_usuario_ts ON transacciones (usuario_id, ts_utc);

-- Cada decisión del sistema: entrada, resultado, versión del modelo y fecha (RF7, RNF7)
CREATE TABLE IF NOT EXISTS predicciones (
    prediccion_id     BIGSERIAL PRIMARY KEY,
    transaccion_id    TEXT NOT NULL,
    usuario_id        TEXT,
    dispositivo_id    TEXT,
    entrada           JSONB NOT NULL,           -- lo que se envió al modelo (sin PAN)
    probabilidad      DOUBLE PRECISION,
    decision          TEXT NOT NULL CHECK (decision IN ('aprobar','revisar','bloquear')),
    origen            TEXT NOT NULL CHECK (origen IN ('modelo','regla_dura','reserva')),
    razones           JSONB NOT NULL,
    version_modelo    TEXT,
    version_umbrales  TEXT,
    latencia_ms       DOUBLE PRECISION,
    creado_utc        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_pred_tx    ON predicciones (transaccion_id);
CREATE INDEX IF NOT EXISTS ix_pred_fecha ON predicciones (creado_utc DESC);
