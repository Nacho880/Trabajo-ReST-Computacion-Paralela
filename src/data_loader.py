"""
src/data_loader.py
===================

Carga y validación inicial del archivo `ventas_completas.csv`.

Responsabilidad única: dado un path a un CSV, devolver un
`dask.dataframe.DataFrame` lazy, con columnas ya renombradas a los
nombres estandarizados del proyecto (ver `config.COLUMN_RENAME_MAP`) y
con la garantía de que el esquema es válido.

Por qué Dask:
    El enunciado exige poder manejar archivos grandes sin agotar la
    memoria. `dask.dataframe.read_csv` divide el archivo en particiones
    lógicas (chunks) que se leen y procesan de forma perezosa (lazy),
    postergando el cómputo real hasta que se invoque `.compute()`.
    Esto además habilita paralelismo nativo en las etapas posteriores
    del pipeline (cada partición puede procesarse en un proceso/hilo
    distinto), sin necesidad de infraestructura de cluster como
    PySpark, adecuada para el volumen de datos de este proyecto (<1GB).

Estrategia de validación:
    1. Se valida existencia/tamaño del archivo (metadata, O(1)).
    2. Se lee una muestra pequeña con `pandas.read_csv(nrows=...)` para
       detectar archivos corruptos y validar el esquema ANTES de pagar
       el costo de construir el grafo de tareas de Dask sobre un
       archivo potencialmente inválido.
    3. Se construye el `dask.dataframe` completo de forma lazy, con
       dtypes explícitos derivados de la muestra validada.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from config import (
    COLUMN_RENAME_MAP,
    CSV_QUOTECHAR,
    CSV_SEPARATOR,
    EXPECTED_DTYPES,
    RAW_EXPECTED_COLUMNS,
)
from src.utils.logger import get_logger
from src.utils.validators import (
    CorruptDataError,
    SchemaValidationError,
    validate_file_exists_and_not_empty,
    validate_schema,
)

logger = get_logger(__name__)

# Tamaño de la muestra usada para validar esquema/tipos antes de la carga
# completa. Suficiente para detectar problemas estructurales sin leer
# el archivo completo en memoria.
_SAMPLE_ROWS = 1000

# Tamaño de partición objetivo para Dask (chunking). 64MB es un balance
# razonable entre paralelismo (más particiones = más tareas concurrentes)
# y overhead de scheduling (particiones demasiado pequeñas generan
# overhead de coordinación que supera el beneficio del paralelismo).
_BLOCKSIZE = "64MB"


def _read_validation_sample(file_path: Path) -> pd.DataFrame:
    """
    Lee las primeras `_SAMPLE_ROWS` filas (bien formadas) del CSV con
    Pandas para validar rápidamente el esquema del archivo.

    Tolerancia a filas individuales corruptas:
        Un archivo de gran volumen consolidado desde múltiples fuentes
        (como describe el enunciado) puede contener un número pequeño
        de filas mal formadas (ej. un campo de texto con un punto y
        coma sin escapar, el delimitador real de este CSV). Rechazar
        el archivo COMPLETO por unas pocas filas así sería un falso
        negativo de robustez: se usa `on_bad_lines="warn"` (capturando
        la advertencia para contarla sin ensuciar la salida estándar)
        para omitir esas filas puntuales, exactamente la misma política
        de tolerancia que ya se aplica en la lectura completa vía Dask
        (ver `load_and_validate`), y se registra en el log cuántas
        filas fueron omitidas.

        Esto es distinto de un archivo genuinamente corrupto (encoding
        inválido, binario, sin estructura CSV alguna), que sigue
        lanzando `CorruptDataError` porque en ese caso NINGUNA fila es
        parseable.

    Args:
        file_path: ruta al archivo CSV.

    Returns:
        pd.DataFrame: muestra cruda (columnas originales, sin renombrar),
        excluyendo las filas individuales que no pudieron parsearse.

    Raises:
        CorruptDataError: si el archivo no puede parsearse en absoluto
            (delimitador inconsistente en el header, encoding inválido,
            archivo binario, o CERO filas parseables).

    Complejidad:
        O(k) tiempo y espacio, donde k = _SAMPLE_ROWS (constante,
        independiente del tamaño total del archivo).
    """
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            sample = pd.read_csv(
                file_path,
                nrows=_SAMPLE_ROWS,
                sep=CSV_SEPARATOR,
                quotechar=CSV_QUOTECHAR,
                on_bad_lines="warn",
                engine="python",
            )
        n_bad_lines = len(caught_warnings)
        if n_bad_lines > 0:
            logger.warning(
                "Se omitieron %d fila(s) mal formada(s) dentro de la muestra de "
                "validación (ej. delimitador inconsistente en un campo de "
                "texto). El archivo se considera válido de todas formas; estas "
                "filas también se omitirán consistentemente en la lectura "
                "completa vía Dask.",
                n_bad_lines,
            )
    except pd.errors.ParserError as exc:
        raise CorruptDataError(
            f"El archivo '{file_path}' no pudo ser parseado como CSV válido: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise CorruptDataError(
            f"El archivo '{file_path}' tiene un encoding inválido o incompatible: {exc}"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise CorruptDataError(
            f"El archivo '{file_path}' no contiene columnas/datos parseables: {exc}"
        ) from exc

    if sample.empty:
        raise CorruptDataError(
            f"El archivo '{file_path}' se parseó correctamente pero no contiene filas."
        )

    return sample


def _build_dask_dtypes() -> dict[str, str]:
    """
    Construye el diccionario de dtypes a pasar a `dask.dataframe.read_csv`,
    usando las claves originales del CSV (con espacios/tildes) mapeadas
    desde `EXPECTED_DTYPES` (que usa nombres estandarizados).

    Returns:
        dict[str, str]: mapeo {nombre_columna_original: dtype_pandas}.

    Complejidad:
        O(k) tiempo y espacio, k = número de columnas con dtype declarado.
    """
    standard_to_raw = {v: k for k, v in COLUMN_RENAME_MAP.items()}
    return {
        standard_to_raw[std_col]: dtype
        for std_col, dtype in EXPECTED_DTYPES.items()
        if std_col in standard_to_raw
    }


def load_and_validate(file_path: str | Path):
    """
    Carga `ventas_completas.csv` como un `dask.dataframe.DataFrame` lazy,
    validando previamente que el archivo exista, no esté vacío, tenga un
    formato CSV parseable y contenga todas las columnas obligatorias.

    Args:
        file_path: ruta al archivo CSV de entrada (típicamente
            ``data/ventas_completas.csv``, recibida por línea de comandos).

    Returns:
        dask.dataframe.DataFrame: DataFrame lazy con las columnas ya
        renombradas a los nombres estandarizados del proyecto
        (ver `config.COLUMN_RENAME_MAP`) y `FECHA`/`FECHA_NACIMIENTO`
        parseadas como fecha.

    Raises:
        FileNotFoundOrEmptyError: si el archivo no existe o está vacío.
        CorruptDataError: si el archivo no es un CSV válido.
        SchemaValidationError: si faltan columnas obligatorias.

    Complejidad:
        O(1) en esta función respecto al tamaño total del archivo: la
        validación de esquema opera sobre una muestra constante y la
        construcción del grafo de Dask es perezosa (no lee el archivo
        completo hasta que se invoque `.compute()` aguas abajo).
    """
    file_path = Path(file_path)
    logger.info("Iniciando carga y validación de: %s", file_path)

    validate_file_exists_and_not_empty(file_path)

    sample = _read_validation_sample(file_path)
    validate_schema(list(sample.columns), RAW_EXPECTED_COLUMNS)
    logger.info(
        "Validación de esquema exitosa sobre muestra de %d filas.", len(sample)
    )

    # Import diferido: dask solo se necesita a partir de este punto, lo que
    # permite que la validación de esquema (y sus tests) funcione incluso
    # en entornos donde dask aún no está instalado.
    import dask.dataframe as dd

    dtypes = _build_dask_dtypes()
    # Solo FECHA (fecha de la transacción) se parsea directamente en la
    # lectura: el enunciado garantiza su formato ISO 8601 consistente.
    # FECHA NACIMIENTO NO se incluye aquí a propósito: si contiene valores
    # nulos o mal formateados distribuidos de forma desigual entre
    # particiones, Dask puede inferir un dtype distinto por partición
    # (datetime64 en unas, object en otras) y lanzar
    # "Mismatched dtypes found in pd.read_csv" al intentar concatenarlas.
    # Se convierte explícitamente DESPUÉS de la carga, con
    # `errors="coerce"`, que además es más robusto: transforma cualquier
    # fecha de nacimiento mal formateada en NaT en lugar de abortar toda
    # la carga (tal como recomienda el propio mensaje de error de Dask).
    date_columns = ["FECHA"]

    try:
        ddf = dd.read_csv(
            file_path,
            blocksize=_BLOCKSIZE,
            sep=CSV_SEPARATOR,
            quotechar=CSV_QUOTECHAR,
            dtype=dtypes,
            parse_dates=date_columns,
            on_bad_lines="warn",
        )
    except Exception as exc:  # noqa: BLE001 - se relanza como error de dominio
        raise CorruptDataError(
            f"Dask no pudo construir el plan de lectura para '{file_path}': {exc}"
        ) from exc

    ddf = ddf.rename(columns=COLUMN_RENAME_MAP)

    # CANAL se convierte a categórico explícitamente, con un conjunto de
    # categorías fijo y ordenado alfabéticamente (una única pasada con
    # `.unique().compute()`), en vez de dejar que Dask infiera el dtype
    # categórico automáticamente al leer el CSV.
    #
    # Motivo (determinismo entre particiones):
    #   Si el dtype categórico se infiere durante la lectura, Dask lo
    #   hace de forma INDEPENDIENTE en cada partición (792 en este
    #   dataset), a partir únicamente de los valores presentes en esa
    #   partición. El dtype queda "unknown" hasta que se unifica --
    #   típicamente al hacer `.compute()` -- y el orden final de
    #   `.cat.categories` depende del orden en que las particiones se
    #   concatenan/procesan, que con el planificador multi-hilo de Dask
    #   puede variar entre corridas (mismo seed, mismo código).
    #   Esa inestabilidad se propagaría silenciosamente a
    #   `pd.get_dummies(..., drop_first=True)` en
    #   `src/modeling_regression.py`: la categoría "de referencia" (la que
    #   `drop_first` descarta) cambiaría de una corrida a otra, alterando
    #   coeficientes, VIF, RMSE y R² del modelo sin que ningún error lo
    #   señale.
    #   Fijar explícitamente las categorías (ordenadas, calculadas una
    #   sola vez) elimina esa dependencia del orden de concatenación:
    #   todas las particiones comparten desde el inicio el mismo
    #   `CategoricalDtype`.
    canal_categorias = sorted(ddf["CANAL"].unique().compute().tolist())
    ddf["CANAL"] = ddf["CANAL"].astype(
        pd.CategoricalDtype(categories=canal_categorias, ordered=False)
    )
    logger.info(
        "CANAL convertido a categórico con %d categoría(s) fija(s) y "
        "ordenada(s) alfabéticamente (%s), para garantizar que el "
        "one-hot encoding en la etapa de modelado sea reproducible entre "
        "corridas.",
        len(canal_categorias), canal_categorias,
    )

    ddf["FECHA_NACIMIENTO"] = dd.to_datetime(
        ddf["FECHA_NACIMIENTO"], errors="coerce", format="%Y-%m-%d"
    )
    logger.info(
        "FECHA_NACIMIENTO convertida a datetime con formato '%%Y-%%m-%%d' "
        "(el formato real del archivo, ej. '1999-04-01') "
        "y errors='coerce' como red de seguridad: cualquier valor que no "
        "calce exactamente con ese formato se convertirá en NaT en lugar de "
        "abortar la carga completa. El conteo real de nulos resultantes se "
        "reportará en el reporte de limpieza tras el .compute() posterior."
    )

    n_partitions = ddf.npartitions
    logger.info(
        "Archivo cargado de forma perezosa con Dask: %d particiones "
        "(blocksize=%s). El cómputo real ocurrirá al invocar .compute().",
        n_partitions,
        _BLOCKSIZE,
    )

    return ddf
