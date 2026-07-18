"""
app/services/dataset.py
=========================

Carga desatendida del CSV de ventas al iniciar la API (sin intervención
manual, tal como exige el enunciado), orquestando el pipeline de Dask:

    src.data_loader.load_and_validate   (carga lazy, chunking, validación de esquema)
    src.data_cleaning.clean_dataset     (nulos + outliers)
    src.feature_engineering.engineer_features  (EDAD, MONTO_POR_UNIDAD, FRECUENCIA_COMPRA, estandarización)

Este módulo se limita a la orquestación mínima necesaria para el startup
de la API y la materialización final a Pandas.

Diseño: Dask se usa de punta a punta para la ETL de arranque (carga +
limpieza + feature engineering). La materialización a Pandas
(`.compute()`) ocurre UNA sola vez, al final, porque el DataFrame
resultante (algunos cientos de MB) cabe cómodamente en memoria y las
consultas HTTP posteriores son operaciones de filtrado de baja latencia
sobre datos ya materializados, para las que reconstruir un grafo de
tareas de Dask en cada request agregaría overhead de scheduling sin
ningún beneficio real.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.db as db
from src.data_cleaning import (
    clean_dataset,
    apply_cleaning_partition,
    compute_global_cleaning_stats,
)
from src.data_loader import load_and_validate
from src.feature_engineering import (
    engineer_features,
    apply_features_partition,
    compute_global_feature_stats,
)
from src.utils.io_utils import save_json
from src.utils.logger import get_logger
from app.services.csv_download import download_csv_if_needed

logger = get_logger(__name__)

# Orden/subconjunto de columnas que se persisten en Postgres (ver
# `sql/ddl.sql`). Los flags de outlier (`<col>_ES_OUTLIER`) y las columnas
# `_STD` de estandarización son subproductos de auditoría del pipeline de
# limpieza/features — se documentan en el log y en los reportes JSON, pero
# no forman parte del contrato de `/v1/estadisticas/ventas`, así que no se
# insertan en `ventas`.
#
# Columnas ORIGINALES del CSV de entrada (ver enunciado, tabla de
# "Estructura de los Archivos CSV"), sin ninguna transformación de
# valor (solo el rename a formato estándar de `config.COLUMN_RENAME_MAP`):
#   FECHA, CANAL, SKU, PRODUCTO, UNIDADES, PORCENTAJE_DESCUENTO,
#   MONTO_APLICADO, BOLETA, LOCAL, CODIGO_CLIENTE, RUN_CLIENTE, NOMBRES,
#   APELLIDOS, FECHA_NACIMIENTO, GENERO.
#
# Columnas CALCULADAS (variables derivadas, generadas por
# `src/feature_engineering.py`, no presentes como tales en el CSV):
#   - MONTO_POR_UNIDAD = MONTO_APLICADO / UNIDADES.
#   - EDAD, calculada a partir de FECHA y FECHA_NACIMIENTO.
#   - FRECUENCIA_COMPRA, conteo de boletas únicas por CODIGO_CLIENTE.
#     Opcional: se omite si `CPYD_ENABLE_FEATURE_ENGINEERING=false` (ver
#     `config.ENABLE_FEATURE_ENGINEERING`), ya que requiere un GROUP BY
#     sobre todo el dataset y puede ser costoso para archivos muy grandes.
_COLUMNAS_INGESTA: tuple[str, ...] = (
    "FECHA", "CANAL", "SKU", "PRODUCTO", "UNIDADES", "PORCENTAJE_DESCUENTO",
    "MONTO_APLICADO", "MONTO_POR_UNIDAD", "BOLETA", "LOCAL", "CODIGO_CLIENTE",
    "RUN_CLIENTE", "NOMBRES", "APELLIDOS", "FECHA_NACIMIENTO", "GENERO",
    "EDAD", "FRECUENCIA_COMPRA",
)


def cargar_dataframe_ventas(csv_path: str | Path) -> pd.DataFrame:
    """
    Ejecuta la ETL completa de arranque y devuelve el DataFrame final
    (Pandas, materializado) listo para servir `/v1/estadisticas/ventas`.

    Args:
        csv_path: ruta al CSV de ventas (típicamente `config.CSV_PATH`).

    Returns:
        pandas.DataFrame: dataset limpio y enriquecido (incluye la
        columna EDAD, requerida por el filtro EDAD del enunciado).

    Raises:
        src.utils.validators.DataValidationError: si el archivo no
            existe ni pudo descargarse, está vacío, es un CSV corrupto
            o le faltan columnas obligatorias. Se deja propagar
            intencionalmente: si el CSV de entrada no es válido, la API
            no debe terminar de arrancar "a medias" sirviendo un
            dataset vacío o parcial.

    Complejidad:
        Dominada por la lectura/parseo del CSV completo,
        O(n) tiempo distribuido entre las particiones de Dask,
        n = número de filas del archivo.
    """
    download_csv_if_needed(csv_path)

    logger.info(">>> INICIO carga desatendida del CSV: %s", csv_path)

    ddf = load_and_validate(csv_path)
    # `.persist()` dispara la lectura completa UNA vez y materializa las
    # particiones en memoria (sin abandonar la API de Dask), evitando
    # que cada `.compute()` parcial de `clean_dataset`/`engineer_features`
    # relea el archivo completo desde cero.
    ddf = ddf.persist()

    ddf_limpio, reporte_limpieza = clean_dataset(ddf)
    ddf_final, parametros_escalamiento = engineer_features(ddf_limpio)

    df = ddf_final.compute()

    logger.info(
        "Dataset de ventas cargado y listo para servir la API: %d filas, "
        "%d columnas, %.2f MB en memoria.",
        df.shape[0], df.shape[1], df.memory_usage(deep=True).sum() / (1024 ** 2),
    )

    # Deja constancia del reporte de limpieza y de los parámetros de
    # estandarización de esta ejecución de la API, como JSON en
    # output/tables/.
    try:
        save_json(reporte_limpieza.to_dict(), "reporte_limpieza_api")
        save_json(parametros_escalamiento.to_dict(), "parametros_estandarizacion_api")
    except OSError:
        # No es crítico para servir la API: si el directorio de salida no
        # es escribible en este entorno, se continúa igualmente (ya
        # quedó registrado el error por `save_json` mismo en el log).
        pass

    return df


# =============================================================================
# Arquitectura de dos pasadas (Dask + Postgres, soporte de hasta ~1.7 TiB)
# =============================================================================
# `cargar_dataframe_ventas` (arriba) requiere que el CSV completo, ya
# limpio y enriquecido, quepa en memoria de un solo proceso
# (`ddf_final.compute()`), algo inviable con un archivo de ~1.7 TiB.
# `app/main.py` usa en su lugar `ingest_if_needed`, que resuelve el mismo
# problema (dejar los datos listos para servir la API) con dos pasadas
# sobre el CSV, sin materializar el dataset completo en memoria en
# ningún punto:
#
#   Pasada 1: `compute_global_cleaning_stats` + `compute_global_feature_stats`
#             recorren el CSV (perezoso) y devuelven ÚNICAMENTE escalares
#             globales (medianas, cotas de outliers, medias/std,
#             frecuencia por cliente). No se inserta nada todavía.
#   Pasada 2: se vuelve a recorrer el CSV, partición por partición
#             (`ddf.to_delayed()`), aplicando esos estadísticos globales
#             (`apply_cleaning_partition` + `apply_features_partition`) y
#             haciendo `COPY` masivo de cada partición a Postgres. Cada
#             partición se libera de memoria (`del pdf`) antes de procesar
#             la siguiente.
#
# La API (`app/routers/ventas.py`) consulta Postgres directamente después:
# nunca vuelve a leer el CSV completo ni mantiene un DataFrame en memoria.
def ingest_if_needed(csv_path: str | Path) -> None:
    """
    Punto de entrada de la carga desatendida con la arquitectura de dos
    pasadas. Idempotente: si `etl_status` ya indica una carga completa
    anterior, no hace nada (evita releer/reinsertar un CSV de 1.7 TiB en
    cada arranque de la API).

    Args:
        csv_path: ruta al CSV de ventas (típicamente `config.CSV_PATH`).

    Raises:
        src.utils.validators.DataValidationError: si el CSV no es válido.
        RuntimeError: si no hay una URL de conexión a Postgres configurada.
    """
    download_csv_if_needed(csv_path)

    conexion = db.get_connection()
    try:
        db.ensure_schema(conexion)

        if db.carga_completa(conexion):
            logger.info(
                "etl_status indica que una carga completa ya se ejecutó "
                "anteriormente. Se omite la ingesta (idempotencia)."
            )
            return

        # Si llegamos aquí, o bien es la primera ingesta, o bien un intento
        # anterior quedó a mitad de camino (p.ej. la conexión/proceso murió,
        # el disco se llenó, etc.). En ese segundo caso, `copy_dataframe`
        # ya hizo COMMIT de las particiones procesadas antes del fallo, así
        # que `ventas` puede contener datos parciales de esa corrida
        # incompleta. Como no hay una clave natural única para deduplicar,
        # se vacía la tabla antes de reprocesar: la única forma de garantizar
        # que un reintento no duplique filas es partir de cero.
        db.reiniciar_estado_carga(conexion)

        logger.info(">>> INICIO ingesta de dos pasadas: %s", csv_path)

        # LAZY: a diferencia de `cargar_dataframe_ventas`, aquí NO se llama
        # `.persist()` — persistir forzaría a tener todas las particiones en
        # memoria simultáneamente, exactamente lo que esta arquitectura evita.
        ddf = load_and_validate(csv_path)

        # --- Pasada 1: solo estadísticas globales, nada se inserta aún ---
        cleaning_stats = compute_global_cleaning_stats(ddf)
        feature_stats = compute_global_feature_stats(ddf)

        # --- Pasada 2: releer por partición, transformar, COPY, liberar ---
        total_filas = 0
        for indice, particion_delayed in enumerate(ddf.to_delayed()):
            pdf = particion_delayed.compute()  # UNA partición, no el dataset completo
            if pdf.empty:
                continue

            pdf = apply_cleaning_partition(pdf, cleaning_stats)
            pdf = apply_features_partition(pdf, feature_stats)

            columnas_presentes = [c for c in _COLUMNAS_INGESTA if c in pdf.columns]
            pdf_db = pdf[columnas_presentes].rename(columns=str.lower)

            # Particiones mensuales creadas dinámicamente a partir de las
            # fechas realmente presentes en ESTE chunk (nunca asumidas).
            db.ensure_month_partitions_for_chunk(conexion, pdf_db, columna_fecha="fecha")
            filas_insertadas = db.copy_dataframe(conexion, pdf_db)
            total_filas += filas_insertadas

            logger.info(
                "Partición %d insertada vía COPY: %d filas (acumulado=%d).",
                indice, filas_insertadas, total_filas,
            )
            del pdf, pdf_db  # libera la partición antes de procesar la siguiente

        db.marcar_carga_completa(conexion)
        logger.info("Ingesta de dos pasadas completa: %d filas totales.", total_filas)
    finally:
        conexion.close()
