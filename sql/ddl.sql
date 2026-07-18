-- sql/ddl.sql
-- ============================================================================
-- Esquema Postgres para "Servicio ReST: Resumen estadístico" (Cruz Morada).
--
-- `ventas` está particionada por RANGE sobre `fecha`. Las particiones
-- mensuales NO se declaran de forma estática en este archivo: se crean
-- dinámicamente en tiempo de ingesta (ver `src/db.py::ensure_month_partition`),
-- una por cada mes que realmente aparece en los datos del CSV que se esté
-- cargando. Este DDL solo define la tabla padre (sin filas propias, tal como
-- exige el particionado declarativo de Postgres) y una partición DEFAULT de
-- respaldo, para que la ingesta nunca falle si por algún motivo una fecha
-- llega antes de que su partición mensual correspondiente exista.
--
-- Es seguro ejecutar este script más de una vez (IF NOT EXISTS / ON CONFLICT
-- en todo lo que crea estado).
-- ============================================================================

CREATE TABLE IF NOT EXISTS ventas (
    fecha                 TIMESTAMP NOT NULL,
    canal                 TEXT,
    sku                   BIGINT,
    producto              TEXT,
    unidades              BIGINT,
    porcentaje_descuento  DOUBLE PRECISION,
    monto_aplicado        DOUBLE PRECISION,
    monto_por_unidad      DOUBLE PRECISION,
    boleta                BIGINT,
    local                 BIGINT,
    codigo_cliente        TEXT,
    run_cliente           TEXT,
    nombres               TEXT,
    apellidos             TEXT,
    fecha_nacimiento      DATE,
    genero                BIGINT,
    edad                  DOUBLE PRECISION,
    frecuencia_compra     BIGINT
) PARTITION BY RANGE (fecha);

-- Partición de respaldo (catch-all). En operación normal debería
-- permanecer vacía: toda fecha real del CSV queda cubierta por su
-- partición mensual, creada dinámicamente ANTES del COPY correspondiente
-- (ver `src/db.py::ensure_month_partitions_for_chunk`).
CREATE TABLE IF NOT EXISTS ventas_default PARTITION OF ventas DEFAULT;

-- Índices secundarios: aceleran los filtros de /v1/estadisticas/ventas
-- (GENERO, CANAL, LOCAL, CODIGO_PRODUCTO, ID_PERSONA, FECHA_DESDE/HASTA).
-- Para volúmenes de millones de filas su costo en espacio es razonable,
-- pero para cargas masivas extremadamente grandes (del orden de TiB),
-- cada índice adicional puede duplicar/triplicar el espacio en disco
-- requerido durante la carga. En ese escenario, considere:
--   1. Crear el esquema con `CPYD_CREATE_INDEXES=false` (ver
--      `src/db.py::_render_ddl`, que omite estas sentencias), cargar los
--      datos, y crear los índices manualmente después de la ingesta
--      (`CREATE INDEX CONCURRENTLY ...` para no bloquear lecturas).
--   2. O crearlos igualmente aquí si el espacio disponible lo permite
--      (comportamiento por defecto, CPYD_CREATE_INDEXES=true).
CREATE INDEX IF NOT EXISTS idx_ventas_fecha           ON ventas (fecha);
CREATE INDEX IF NOT EXISTS idx_ventas_genero          ON ventas (genero);
CREATE INDEX IF NOT EXISTS idx_ventas_canal           ON ventas (canal);
CREATE INDEX IF NOT EXISTS idx_ventas_local           ON ventas (local);
CREATE INDEX IF NOT EXISTS idx_ventas_sku             ON ventas (sku);
CREATE INDEX IF NOT EXISTS idx_ventas_codigo_cliente  ON ventas (codigo_cliente);

-- ----------------------------------------------------------------------------
-- etl_status: marcador MÍNIMO de finalización de una carga completa.
--
-- Una única fila (forzado por la restricción `id BOOLEAN ... CHECK (id)`,
-- que solo admite el valor TRUE como clave primaria). `completed = TRUE`
-- significa exclusivamente "una ejecución de la Pasada 2 terminó sin
-- excepciones de principio a fin"; nunca se marca a mitad de camino.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS etl_status (
    id           BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    completed    BOOLEAN NOT NULL DEFAULT FALSE,
    completed_at TIMESTAMPTZ
);

INSERT INTO etl_status (id, completed, completed_at)
VALUES (TRUE, FALSE, NULL)
ON CONFLICT (id) DO NOTHING;
