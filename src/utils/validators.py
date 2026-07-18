"""
src/utils/validators.py
========================

Excepciones de dominio y funciones de validación reutilizables.

Estas validaciones se usan principalmente en `data_loader.py`, pero se
mantienen aquí (y no allí) para poder reutilizarlas en `tests/` sin
depender de la lógica de carga completa (principio de responsabilidad
única y DRY).

Todas las funciones operan sobre estructuras estándar de Python/Pandas
(no requieren un objeto Dask materializado), de modo que puedan
aplicarse tanto sobre una partición pequeña (para validar rápido antes
de leer el archivo completo) como sobre el DataFrame final ya
computado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Excepciones de dominio
# ---------------------------------------------------------------------------
class DataValidationError(Exception):
    """Excepción base para todo error de validación de datos del proyecto."""


class FileNotFoundOrEmptyError(DataValidationError):
    """Se lanza cuando el archivo de entrada no existe o está vacío."""


class SchemaValidationError(DataValidationError):
    """Se lanza cuando faltan columnas esperadas o los tipos no son compatibles."""


class CorruptDataError(DataValidationError):
    """Se lanza cuando el archivo no puede ser parseado como CSV válido."""


class CSVDownloadError(DataValidationError):
    """
    Se lanza cuando el CSV no existe localmente y su descarga automática
    (ver `app/services/csv_download.py::download_csv_if_needed`) falla:
    sin URL configurada, error de red, código HTTP de error, timeout, o
    archivo descargado vacío/corrupto.
    """


# ---------------------------------------------------------------------------
# Validaciones de archivo
# ---------------------------------------------------------------------------
def validate_file_exists_and_not_empty(file_path: Path) -> None:
    """
    Verifica que el archivo exista y no esté vacío (0 bytes).

    Args:
        file_path: ruta al archivo CSV de entrada.

    Raises:
        FileNotFoundOrEmptyError: si el archivo no existe, no es un archivo
            regular, o su tamaño es 0 bytes.

    Complejidad:
        O(1) tiempo y espacio (solo se consulta metadata del filesystem).
    """
    if not file_path.exists():
        raise FileNotFoundOrEmptyError(
            f"El archivo de entrada no existe: '{file_path}'."
        )
    if not file_path.is_file():
        raise FileNotFoundOrEmptyError(
            f"La ruta indicada no corresponde a un archivo: '{file_path}'."
        )
    if file_path.stat().st_size == 0:
        raise FileNotFoundOrEmptyError(
            f"El archivo de entrada está vacío (0 bytes): '{file_path}'."
        )


# ---------------------------------------------------------------------------
# Validaciones de esquema
# ---------------------------------------------------------------------------
def validate_schema(df_columns: list[str], expected_columns: tuple[str, ...]) -> None:
    """
    Verifica que todas las columnas esperadas estén presentes.

    No exige orden ni ausencia de columnas extra: solo que el conjunto de
    columnas esperadas sea subconjunto de las columnas reales. Esto hace la
    validación robusta ante reordenamientos o columnas adicionales futuras.

    Args:
        df_columns: lista de nombres de columnas presentes en los datos.
        expected_columns: tupla de nombres de columnas requeridas.

    Raises:
        SchemaValidationError: si falta una o más columnas esperadas.

    Complejidad:
        O(n + m) tiempo, O(n + m) espacio, donde n = len(df_columns),
        m = len(expected_columns) (uso de sets para la comparación).
    """
    actual_set = set(df_columns)
    expected_set = set(expected_columns)
    missing = expected_set - actual_set

    if missing:
        raise SchemaValidationError(
            "Faltan columnas obligatorias en el archivo de entrada: "
            f"{sorted(missing)}. Columnas encontradas: {sorted(actual_set)}."
        )


def validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """
    Verifica que una columna exista y sea de tipo numérico.

    Args:
        df: DataFrame (Pandas) a validar. Al provenir de un `.compute()`
            de Dask o de una partición pequeña, la interfaz es idéntica.
        column: nombre de la columna a verificar.

    Raises:
        SchemaValidationError: si la columna no existe o no es numérica.

    Complejidad:
        O(1) tiempo y espacio (solo se consulta el dtype, no los datos).
    """
    if column not in df.columns:
        raise SchemaValidationError(f"La columna '{column}' no existe en el DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise SchemaValidationError(
            f"La columna '{column}' debería ser numérica pero tiene tipo "
            f"{df[column].dtype}."
        )


def check_no_infinite_values(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, int]:
    """
    Cuenta valores infinitos (+inf / -inf) por columna numérica.

    No lanza excepción: los valores infinitos se reportan para que el
    módulo de limpieza (`data_cleaning.py`) decida cómo tratarlos
    (habitualmente como un caso especial de outlier).

    Args:
        df: DataFrame a inspeccionar.
        numeric_columns: columnas numéricas a revisar.

    Returns:
        dict[str, int]: mapeo columna -> cantidad de valores infinitos.

    Complejidad:
        O(n * k) tiempo, donde n = número de filas, k = número de columnas
        revisadas; O(k) espacio para el resultado.
    """
    result: dict[str, int] = {}
    for column in numeric_columns:
        if column not in df.columns:
            continue
        count = int(np.isinf(df[column].to_numpy(dtype="float64", na_value=0.0)).sum())
        result[column] = count
        if count > 0:
            logger.warning("Columna '%s' contiene %d valores infinitos.", column, count)
    return result
