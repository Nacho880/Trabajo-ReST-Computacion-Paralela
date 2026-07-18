"""
src/db.py
=========

Capa de acceso a Postgres para la arquitectura de dos pasadas:

    - Conexión (`get_connection`).
    - Aplicación del esquema (`ensure_schema`, ejecuta `sql/ddl.sql`).
    - Particionado DINÁMICO de `ventas` por mes calendario, calculado a
      partir de las fechas que realmente aparecen en cada chunk que se va
      a insertar (`ensure_month_partition`, `ensure_month_partitions_for_chunk`)
      — nunca se asume de antemano qué meses trae el CSV.
    - Carga masiva por partición vía `COPY` (`copy_dataframe`).
    - `etl_status` en su versión mínima (`carga_completa`,
      `marcar_carga_completa`, `reiniciar_estado_carga`).
    - Ejecución de la consulta agregada que responde
      `/v1/estadisticas/ventas` (`ejecutar_resumen`).

Ninguna función de este módulo materializa más de una partición/chunk a
la vez en memoria: el llamador (`app/services/dataset.py`) es quien
decide el tamaño de cada chunk que se pasa a `copy_dataframe`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DDL_PATH = Path(__file__).resolve().parent.parent / "sql" / "ddl.sql"


def get_connection(database_url: str | None = None):
    """
    Abre una conexión a Postgres usando `psycopg2`.

    Args:
        database_url: cadena de conexión. Si no se provee, usa
            `config.DATABASE_URL`.

    Raises:
        RuntimeError: si no hay ninguna URL de conexión configurada.
    """
    import psycopg2  # import diferido: no requerido para usar el resto del proyecto sin Postgres.

    url = database_url or config.DATABASE_URL
    if not url:
        raise RuntimeError(
            "\n"
            "No existe una conexión PostgreSQL configurada.\n"
            "Configure la variable de entorno CPYD_DATABASE_URL en el "
            "archivo .env (ver .env.example).\n"
            "Ejemplo:\n"
            "    postgresql://usuario:password@host:5432/base_datos\n"
        )
    return psycopg2.connect(url)


def ensure_schema(conn) -> None:
    """
    Aplica `sql/ddl.sql` (idempotente: seguro de ejecutar más de una vez).

    Crea `ventas`, `ventas_default` y `etl_status` si no existen
    (`CREATE TABLE IF NOT EXISTS`). Si `CPYD_CREATE_INDEXES=false`, se
    omiten los `CREATE INDEX` (ver `_render_ddl`): útil para cargas muy
    grandes donde los índices adicionales pueden duplicar/triplicar el
    espacio en disco requerido y conviene crearlos manualmente después
    de la ingesta.
    """
    ddl_sql = _render_ddl()
    with conn.cursor() as cur:
        cur.execute(ddl_sql)
    conn.commit()
    logger.info("Esquema Postgres verificado/aplicado (%s).", _DDL_PATH.name)


def _render_ddl() -> str:
    """
    Lee `sql/ddl.sql` y, si `config.CREATE_INDEXES` es `False`, elimina
    las sentencias `CREATE INDEX` antes de ejecutarlo (ver
    `CPYD_CREATE_INDEXES` en `.env.example`).
    """
    ddl_sql = _DDL_PATH.read_text(encoding="utf-8")
    if config.CREATE_INDEXES:
        return ddl_sql
    lineas = [
        linea for linea in ddl_sql.splitlines()
        if not linea.strip().upper().startswith("CREATE INDEX")
    ]
    logger.info(
        "CPYD_CREATE_INDEXES=false: se omite la creación de índices "
        "secundarios de 'ventas' durante ensure_schema()."
    )
    return "\n".join(lineas)


def ensure_schema_ready() -> None:
    """
    Abre una conexión Postgres, aplica el esquema (`ensure_schema`) y la
    cierra. Punto de entrada explícito usado por `app/main.py` para dejar
    visible, en el flujo de arranque, el paso
    "¿existe la tabla ventas? -> si no, crear esquema" antes de decidir si
    hace falta cargar el CSV.

    Raises:
        RuntimeError: si no hay una URL de conexión Postgres configurada
            (ver `get_connection`).
    """
    conexion = get_connection()
    try:
        ensure_schema(conexion)
    finally:
        conexion.close()


# ---------------------------------------------------------------------------
# Particionado dinámico por mes
# ---------------------------------------------------------------------------
def nombre_particion_para_fecha(fecha) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    """
    Calcula el nombre de la partición mensual y su rango [inicio, fin)
    para una fecha dada, SIN asumir ningún mes específico: se deriva por
    completo del valor de `fecha` recibido.

    Función pura (sin efectos de red/DB), para que el cálculo del nombre
    y del rango sea testeable sin una conexión real.

    Args:
        fecha: cualquier valor convertible con `pandas.Timestamp`.

    Returns:
        tuple[str, pd.Timestamp, pd.Timestamp]: (nombre_particion,
        inicio_inclusive, fin_exclusivo).
    """
    ts = pd.Timestamp(fecha)
    inicio = pd.Timestamp(year=ts.year, month=ts.month, day=1)
    fin = inicio + pd.DateOffset(months=1)
    nombre = f"ventas_y{inicio.year:04d}m{inicio.month:02d}"
    return nombre, inicio, fin


def ensure_month_partition(conn, fecha) -> str:
    """
    Crea (si no existe) la partición mensual de `ventas` que corresponde
    a `fecha`. Idempotente (`IF NOT EXISTS`).

    Returns:
        str: nombre de la partición.
    """
    nombre, inicio, fin = nombre_particion_para_fecha(fecha)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {nombre}
            PARTITION OF {config.VENTAS_TABLE}
            FOR VALUES FROM (%s) TO (%s)
            """,
            (inicio.to_pydatetime(), fin.to_pydatetime()),
        )
    conn.commit()
    return nombre


def ensure_month_partitions_for_chunk(
    conn, df: pd.DataFrame, columna_fecha: str = "FECHA"
) -> list[str]:
    """
    Crea todas las particiones mensuales necesarias para las fechas
    PRESENTES en `df` (un chunk/partición de Dask ya materializado),
    calculadas dinámicamente a partir de los datos, nunca de una lista
    fija de meses.

    Un único chunk puede abarcar más de un mes calendario: se crean
    todas las particiones que hagan falta antes del `COPY`.

    Returns:
        list[str]: nombres de las particiones creadas/verificadas.
    """
    if df.empty:
        return []
    meses_presentes = pd.to_datetime(df[columna_fecha], errors="coerce").dropna()
    if meses_presentes.empty:
        return []
    periodos = meses_presentes.dt.to_period("M").unique()
    nombres = [ensure_month_partition(conn, periodo.start_time) for periodo in periodos]
    return nombres


# ---------------------------------------------------------------------------
# Carga masiva (COPY)
# ---------------------------------------------------------------------------
def copy_dataframe(conn, df: pd.DataFrame, table: str | None = None) -> int:
    """
    Inserta `df` en `table` (por defecto `config.VENTAS_TABLE`) mediante
    `COPY ... FROM STDIN`, el mecanismo de carga masiva de Postgres.

    Las columnas de `df` deben venir ya en minúsculas (nombres de columna
    de Postgres); usar `df.rename(columns=str.lower)` antes de llamar si
    vienen en el formato estándar del proyecto (mayúsculas).

    Antes de llamar a esta función el llamador debe garantizar (vía
    `ensure_month_partitions_for_chunk`) que existan las particiones
    mensuales necesarias para las fechas de `df`; si alguna fecha queda
    fuera de cualquier partición mensual, cae en `ventas_default`.

    Args:
        conn: conexión Postgres abierta.
        df: chunk ya limpio/enriquecido, con columnas en minúsculas.
        table: tabla destino.

    Returns:
        int: número de filas insertadas (`len(df)`, 0 si `df` está vacío).
    """
    if df.empty:
        return 0
    table = table or config.VENTAS_TABLE

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)

    import psycopg2  # import diferido, mismo motivo que en get_connection().

    columnas_sql = ", ".join(df.columns)
    try:
        with conn.cursor() as cur:
            cur.copy_expert(
                f"COPY {table} ({columnas_sql}) FROM STDIN WITH (FORMAT csv, NULL '')",
                buffer,
            )
        conn.commit()
    except psycopg2.errors.DiskFull:
        # Deja el mensaje explícito ANTES de propagar: sin esto, el error
        # original de psycopg2/libpq (a veces solo "could not extend file"
        # o el genérico "DiskFull") no deja claro que la causa es espacio
        # insuficiente en el servidor PostgreSQL, no un problema de datos.
        conn.rollback()
        raise RuntimeError(
            "La base de datos no tiene espacio suficiente para almacenar "
            "el dataset. Libere espacio en la instancia PostgreSQL o "
            "amplíe su capacidad de almacenamiento antes de reintentar la "
            "carga."
        ) from None
    return len(df)


# ---------------------------------------------------------------------------
# etl_status (versión mínima: una fila, completed + completed_at)
# ---------------------------------------------------------------------------
def carga_completa(conn) -> bool:
    """True si una carga anterior terminó sin interrupciones."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT completed FROM {config.ETL_STATUS_TABLE} WHERE id = TRUE")
        fila = cur.fetchone()
    return bool(fila and fila[0])


def marcar_carga_completa(conn) -> None:
    """Marca la carga como completa. Se llama SOLO tras procesar la última partición sin errores."""
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {config.ETL_STATUS_TABLE} "
            "SET completed = TRUE, completed_at = now() WHERE id = TRUE"
        )
    conn.commit()
    logger.info("etl_status marcado como completado.")


def reiniciar_estado_carga(conn) -> None:
    """
    Vacía `ventas` y vuelve a marcar la carga como incompleta.

    Uso: pruebas de integración y reprocesos manuales explícitos. NUNCA
    se llama desde el flujo normal de arranque de la API.
    """
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {config.VENTAS_TABLE}")
        cur.execute(
            f"UPDATE {config.ETL_STATUS_TABLE} "
            "SET completed = FALSE, completed_at = NULL WHERE id = TRUE"
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Consulta agregada de `/v1/estadisticas/ventas`
# ---------------------------------------------------------------------------
def ejecutar_resumen(conn, columna: str, where_sql: str, params: list) -> dict:
    """
    Ejecuta la consulta agregada (SQL parametrizado) que responde el
    resumen estadístico de `columna`, filtrada por `where_sql`/`params`
    (construidos por `app/services/estadisticas.py::construir_clausula_where`).

    Fórmulas según el enunciado:
        - promedio = suma / conteo (calculado en Python, no en SQL).
        - mediana: `PERCENTILE_CONT(0.5)` (interpola entre los dos
          valores centrales cuando el conteo es par, tal como exige el
          enunciado).
        - desviación estándar: muestral (`STDDEV_SAMP`, ddof=1).

    Returns:
        dict: las 7 claves exactas del contrato de la API. Todo en 0.0 /
        0 si no hay filas que cumplan el filtro.
    """
    condicion = f"WHERE {columna} IS NOT NULL"
    if where_sql:
        condicion += f" AND {where_sql}"

    consulta = f"""
        SELECT
            COALESCE(SUM({columna}), 0)          AS suma,
            COUNT({columna})                      AS conteo,
            MIN({columna})                        AS minimo,
            MAX({columna})                        AS maximo,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {columna}) AS mediana,
            STDDEV_SAMP({columna})                AS desviacion_estandar
        FROM {config.VENTAS_TABLE}
        {condicion}
    """
    with conn.cursor() as cur:
        cur.execute(consulta, params)
        suma, conteo, minimo, maximo, mediana, desviacion = cur.fetchone()

    conteo = int(conteo or 0)
    if conteo == 0:
        return {
            "suma": 0.0, "conteo": 0, "promedio": 0.0, "minimo": 0.0,
            "maximo": 0.0, "mediana": 0.0, "desviacion_estandar": 0.0,
        }

    suma = float(suma)
    return {
        "suma": suma,
        "conteo": conteo,
        "promedio": suma / conteo,
        "minimo": float(minimo),
        "maximo": float(maximo),
        "mediana": float(mediana),
        "desviacion_estandar": float(desviacion) if desviacion is not None else 0.0,
    }
