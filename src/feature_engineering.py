"""
src/feature_engineering.py
============================

Creación de variables derivadas y normalización/estandarización.

Responsabilidad única: a partir del `dask.dataframe.DataFrame` ya
limpio (salida de `data_cleaning.clean_dataset`, aún perezoso pero
persistido en memoria desde la carga), generar las variables
solicitadas explícitamente por el enunciado:

    - MONTO_POR_UNIDAD = MONTO_APLICADO / UNIDADES
    - EDAD (a partir de FECHA_NACIMIENTO y FECHA de la transacción)
    - FRECUENCIA_COMPRA por cliente (conteo de boletas por CODIGO_CLIENTE)

y estandarizar variables numéricas cuando corresponda, documentando
explícitamente los parámetros (media y desviación estándar) usados,
tal como exige el enunciado.

Diseño: Dask de punta a punta, igual que `data_cleaning.py`. Las
transformaciones de columnas permanecen perezosas; solo se
materializan (`.compute()`) los escalares (medias, medianas,
conteos) necesarios para el log/reporte y para decisiones de control
de flujo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import dask.dataframe as dd
import numpy as np
import pandas as pd

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScalingParameters:
    """
    Parámetros de estandarización (media, desviación estándar) usados
    por columna, para que puedan documentarse en el informe técnico y
    reutilizarse de forma determinista sobre datos nuevos (ej. al
    aplicar el mismo modelo a un conjunto de test).

    Attributes:
        means: media muestral usada por columna.
        stds: desviación estándar muestral (ddof=1) usada por columna.
    """

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializa los parámetros a un diccionario plano apto para JSON."""
        return {"means": self.means, "stds": self.stds}


def add_monto_por_unidad(df: dd.DataFrame) -> dd.DataFrame:
    """
    Crea la variable derivada MONTO_POR_UNIDAD = MONTO_APLICADO / UNIDADES.

    Maneja explícitamente el caso UNIDADES=0 (división por cero), que en
    un sistema de ventas real puede ocurrir por errores de registro:
    en ese caso, MONTO_POR_UNIDAD se define como NaN (no como inf),
    dejando constancia en el log de cuántos casos se vieron afectados.
    `.mask(condicion)` reemplaza con NaN donde la condición es True,
    manteniendo la transformación perezosa.

    Args:
        df: `dask.dataframe.DataFrame` limpio con columnas
            MONTO_APLICADO y UNIDADES.

    Returns:
        dd.DataFrame: DataFrame perezoso con la columna
        MONTO_POR_UNIDAD agregada (transformación encolada).

    Complejidad:
        O(n) tiempo distribuido para el conteo de filas con
        UNIDADES=0 (único agregado materializado); O(1) para encolar
        la división elemento a elemento.
    """
    zero_units_mask = df["UNIDADES"] == 0
    n_zero_units = int(zero_units_mask.sum().compute())
    if n_zero_units > 0:
        logger.warning(
            "%d filas tienen UNIDADES=0; MONTO_POR_UNIDAD se define como NaN "
            "para evitar división por cero (posibles errores de registro).",
            n_zero_units,
        )

    # UNIDADES se castea a float64 (dtype nativo de NumPy) antes de dividir:
    # dividir MONTO_APLICADO (float64) por una columna con dtype nullable de
    # Pandas (ej. "Int64", usada en UNIDADES) produce un resultado que
    # también hereda ese dtype nullable ("Float64", con F mayúscula), lo
    # que rompe operaciones de Dask basadas en cuantiles más adelante
    # (mismo motivo documentado en data_cleaning.py). float64 preserva
    # pd.NA -> NaN sin alterar el dtype original de `df["UNIDADES"]`.
    safe_units = df["UNIDADES"].mask(zero_units_mask).astype("float64")
    df["MONTO_POR_UNIDAD"] = df["MONTO_APLICADO"] / safe_units

    mean_value = float(df["MONTO_POR_UNIDAD"].mean(skipna=True).compute())
    median_value = float(df["MONTO_POR_UNIDAD"].quantile(0.5).compute())
    logger.info(
        "Variable MONTO_POR_UNIDAD creada. Media=%.2f, Mediana=%.2f",
        mean_value, median_value,
    )
    return df


def add_edad(df: dd.DataFrame) -> dd.DataFrame:
    """
    Crea la variable derivada EDAD, calculada como la cantidad de años
    calendario completos transcurridos entre FECHA_NACIMIENTO y FECHA
    (fecha de la transacción), NO respecto a la fecha actual de
    ejecución del programa.

    Cálculo por componentes año/mes/día (no por resta de fechas):
        Restar directamente dos columnas datetime64 construye un
        Timedelta cuya representación interna en nanosegundos (int64)
        solo cubre aproximadamente ±292 años. Una FECHA_NACIMIENTO con
        un error de captura muy antiguo puede hacer que esa diferencia
        desborde (OverflowError), como se verificó empíricamente con
        el archivo real de este proyecto. El cálculo por componentes
        opera sobre enteros pequeños y es inmune a ese desbordamiento.
        El accessor `.dt` funciona idéntico en Dask y Pandas.

    Args:
        df: `dask.dataframe.DataFrame` con columnas FECHA y
            FECHA_NACIMIENTO (datetime).

    Returns:
        dd.DataFrame: DataFrame perezoso con la columna EDAD agregada.

    Complejidad:
        O(n) tiempo distribuido; los agregados materializados (conteo
        de inválidos, media, conteo de válidos) son reducciones O(n)
        adicionales sobre la misma columna ya calculada.
    """
    years_diff = df["FECHA"].dt.year - df["FECHA_NACIMIENTO"].dt.year
    birthday_not_yet_occurred = (
        (df["FECHA"].dt.month < df["FECHA_NACIMIENTO"].dt.month)
        | (
            (df["FECHA"].dt.month == df["FECHA_NACIMIENTO"].dt.month)
            & (df["FECHA"].dt.day < df["FECHA_NACIMIENTO"].dt.day)
        )
    )
    df["EDAD"] = (years_diff - birthday_not_yet_occurred.astype("Int64")).astype("float64")

    # Edades negativas o absurdamente altas (>110 años) se tratan como
    # inconsistencias de registro, no como outliers de negocio: se
    # convierten a NaN explícitamente, no se eliminan filas.
    invalid_age_mask = (df["EDAD"] < 0) | (df["EDAD"] > 110)
    n_invalid = int(invalid_age_mask.sum().compute())
    if n_invalid > 0:
        logger.warning(
            "%d filas con EDAD fuera de rango válido [0, 110] fueron "
            "convertidas a NaN (inconsistencia de FECHA_NACIMIENTO).",
            n_invalid,
        )
        df["EDAD"] = df["EDAD"].mask(invalid_age_mask)

    mean_edad = float(df["EDAD"].mean(skipna=True).compute())
    # Se usa `~isna()` en lugar de `.notna()`: en algunas versiones del
    # backend query-optimizado de Dask (dask_expr), `.notna()` sobre una
    # Series no está correctamente expuesto y lanza
    # "AttributeError: 'Series' object has no attribute 'notna'".
    # `~isna()` es equivalente y está soportado de forma estable en todas
    # las versiones de dask.dataframe.
    n_validos = int((~df["EDAD"].isna()).sum().compute())
    n_total = len(df)
    logger.info(
        "Variable EDAD creada. Media=%.1f años (n válidos=%d de %d).",
        mean_edad, n_validos, n_total,
    )
    return df


def add_frecuencia_compra(df: dd.DataFrame) -> dd.DataFrame:
    """
    Crea la variable derivada FRECUENCIA_COMPRA: número total de
    boletas (transacciones) distintas asociadas a cada CODIGO_CLIENTE
    en todo el período cubierto por el dataset.

    Se calcula sobre BOLETA únicas (no sobre número de filas), ya que
    una misma boleta puede tener múltiples líneas de producto y no
    debe contarse como más de una compra.

    `groupby().nunique()` es una operación nativa de `dask.dataframe`
    (paralela y perezosa). El resultado se une a `df` mediante un
    broadcast join mapeado por partición (ver más abajo) en lugar de
    `.merge()`.

    Por qué NO usar `.merge()` aquí (determinismo del orden de filas):
        `df.merge(..., on="CODIGO_CLIENTE")` en `dask.dataframe` requiere
        un *shuffle* interno: las filas se redistribuyen entre
        particiones según el hash de la clave de unión para poder
        emparejarlas. El orden final de las filas tras ese shuffle NO
        está garantizado idéntico entre corridas -- con el planificador
        multi-hilo de Dask, el orden en que las tareas terminan puede
        variar de una ejecución a otra, incluso con el mismo código y
        semilla. Esa reordenación silenciosa de filas se propagaría aguas
        abajo a cualquier muestreo posterior con `random_state` fijo (ej.
        el test de Kolmogorov-Smirnov de `eda.py`, o el muestreo de
        residuos en `modeling_regression.py`): la semilla seguiría siendo
        la misma, pero al aplicarse sobre filas en un orden distinto, el
        resultado del muestreo "reproducible" terminaría siendo distinto
        en cada corrida.

        Como `CODIGO_CLIENTE` es una dimensión de cliente (cardinalidad
        muchísimo menor que el número de transacciones), es seguro y
        eficiente calcular `frecuencia_por_cliente` una sola vez,
        traerla a memoria como una Serie de pandas (`.compute()`), y
        usarla como tabla de lookup vía `Series.map(...)`. `.map()` se
        aplica partición por partición sin ningún shuffle, preservando
        exactamente el orden de filas original de `df`.

    Args:
        df: `dask.dataframe.DataFrame` con columnas CODIGO_CLIENTE y
            BOLETA.

    Returns:
        dd.DataFrame: DataFrame perezoso con la columna
        FRECUENCIA_COMPRA agregada, en el mismo orden de filas que la
        entrada.

    Complejidad:
        O(n) tiempo distribuido para el groupby, O(k) tiempo/espacio
        para el `.map()` de broadcast (k = número de clientes únicos,
        que cabe cómodamente en memoria de un solo proceso).
    """
    frecuencia_por_cliente = (
        df.groupby("CODIGO_CLIENTE")["BOLETA"]
        .nunique()
        .rename("FRECUENCIA_COMPRA")
    )

    # Se materializa una única vez como Serie de pandas (índice =
    # CODIGO_CLIENTE) y se usa como tabla de lookup: `.map()` sobre una
    # Serie de Dask con un mapeo pandas/dict se aplica por partición, sin
    # shuffle, preservando el orden original de `df`.
    frecuencia_map = frecuencia_por_cliente.compute()
    # `meta=` se especifica explícitamente para evitar que Dask intente
    # adivinar el dtype de salida ejecutando la función sobre una muestra
    # pequeña (lo que además genera un UserWarning). `nunique()` siempre
    # devuelve enteros, por lo que el dtype correcto es "int64" (no el
    # "float64" que Dask adivina por defecto al no tener más contexto).
    df["FRECUENCIA_COMPRA"] = df["CODIGO_CLIENTE"].map(
        frecuencia_map, meta=("FRECUENCIA_COMPRA", "int64")
    )

    mean_freq = float(frecuencia_map.mean())
    max_freq = int(frecuencia_map.max())
    n_clientes = len(frecuencia_map)
    logger.info(
        "Variable FRECUENCIA_COMPRA creada para %d clientes únicos. "
        "Media=%.2f compras/cliente, máximo=%d.",
        n_clientes, mean_freq, max_freq,
    )
    return df


def standardize_columns(
    df: dd.DataFrame, columns: list[str]
) -> tuple[dd.DataFrame, ScalingParameters]:
    """
    Estandariza (z-score) las columnas indicadas: x' = (x - media) / std.

    Los parámetros de escalamiento (media, std) se calculan sobre los
    datos provistos y se retornan explícitamente en `ScalingParameters`
    para su documentación en el informe técnico, tal como exige el
    enunciado ("documentando los parámetros utilizados").

    Columnas constantes (std=0) se dejan sin transformar (se documenta
    std=0 en el reporte) para evitar división por cero.

    Args:
        df: `dask.dataframe.DataFrame` con las columnas numéricas a
            estandarizar.
        columns: lista de nombres de columnas a transformar.

    Returns:
        tuple[dd.DataFrame, ScalingParameters]: DataFrame perezoso con
        columnas `<col>_STD` agregadas donde corresponda, y los
        parámetros de escalamiento usados.

    Complejidad:
        O(n * k) tiempo distribuido, k = len(columns); O(1) espacio
        adicional mientras las transformaciones se mantengan perezosas.
    """
    params = ScalingParameters()

    for column in columns:
        if column not in df.columns:
            logger.warning(
                "Columna '%s' no encontrada; se omite en la estandarización.",
                column,
            )
            continue

        # Cast a float64 (dtype nativo de NumPy): columnas con dtype
        # nullable de Pandas (ej. "Int64", caso de UNIDADES) propagan ese
        # mismo tipo nullable ("Float64", con F mayúscula) al restar/dividir,
        # lo que rompe operaciones de Dask basadas en cuantiles más
        # adelante en el pipeline. float64 preserva pd.NA -> NaN sin
        # alterar el dtype original de `df[column]`.
        series = df[column].astype("float64")
        mean = float(series.mean(skipna=True).compute())
        std = float(series.std(skipna=True, ddof=1).compute())
        params.means[column] = mean
        params.stds[column] = std

        std_column = f"{column}_STD"
        if std == 0 or np.isnan(std):
            # Se omite deliberadamente la creación de `<col>_STD`: una
            # copia sin transformar bajo ese nombre daría la falsa
            # impresión de que la variable fue estandarizada cuando en
            # realidad es constante y no tiene sentido estadístico
            # hacerlo. Se documenta la decisión en el log y en
            # `ScalingParameters` (con std=0), pero no se agrega la
            # columna al DataFrame.
            logger.warning(
                "Columna '%s' tiene desviación estándar 0 o indefinida "
                "(variable constante): NO se crea '%s' para evitar dar la "
                "falsa impresión de una estandarización válida. La media/std "
                "quedan igualmente documentadas en el reporte de parámetros.",
                column, std_column,
            )
            continue

        df[std_column] = (series - mean) / std
        logger.info(
            "Columna '%s' estandarizada -> '%s' (media=%.4f, std=%.4f).",
            column, std_column, mean, std,
        )

    return df, params


def engineer_features(
    df: dd.DataFrame, include_frecuencia_compra: bool | None = None
) -> tuple[dd.DataFrame, ScalingParameters]:
    """
    Orquesta la creación de todas las variables derivadas exigidas por
    el enunciado y la estandarización de las variables numéricas
    relevantes, de forma perezosa sobre un `dask.dataframe.DataFrame`.

    El DataFrame retornado NO ha sido materializado a Pandas: la
    materialización final (`.compute()`) ocurre después, marcando la
    frontera entre el procesamiento masivo (Dask: carga + limpieza +
    feature engineering) y el filtrado/cálculo de estadísticas sobre
    requests HTTP (Pandas, ya en memoria).

    Args:
        df: `dask.dataframe.DataFrame` limpio (salida de
            `data_cleaning.clean_dataset`), aún perezoso.
        include_frecuencia_compra: si se calcula FRECUENCIA_COMPRA (el
            `groupby("CODIGO_CLIENTE")` más costoso del pipeline, ya que
            recorre el dataset completo). Si es `None` (por defecto), se
            usa `config.ENABLE_FEATURE_ENGINEERING` (variable de entorno
            `CPYD_ENABLE_FEATURE_ENGINEERING`). Poner en `False` para
            archivos muy grandes donde conviene separar la carga mínima
            (CSV -> BD) del cálculo de variables avanzadas.

    Returns:
        tuple[dd.DataFrame, ScalingParameters]: DataFrame perezoso
        enriquecido con MONTO_POR_UNIDAD, EDAD, opcionalmente
        FRECUENCIA_COMPRA, y columnas `_STD`, junto a los parámetros de
        escalamiento usados.

    Complejidad:
        O(n log n) tiempo distribuido si se incluye
        `add_frecuencia_compra` (dominado por su groupby); O(n) si se
        omite. O(1) espacio adicional mientras el grafo de tareas se
        mantenga perezoso.
    """
    if include_frecuencia_compra is None:
        include_frecuencia_compra = config.ENABLE_FEATURE_ENGINEERING

    df = add_monto_por_unidad(df)
    df = add_edad(df)
    if include_frecuencia_compra:
        df = add_frecuencia_compra(df)
    else:
        logger.info(
            "CPYD_ENABLE_FEATURE_ENGINEERING=false: se omite el cálculo "
            "de FRECUENCIA_COMPRA (groupby por CODIGO_CLIENTE)."
        )

    columns_to_scale = ["MONTO_APLICADO", "UNIDADES", "PORCENTAJE_DESCUENTO"]
    df, scaling_params = standardize_columns(df, columns_to_scale)

    logger.info(
        "Ingeniería de variables (Dask, perezosa) encolada. Columnas totales: %d.",
        len(df.columns),
    )
    return df, scaling_params


# =============================================================================
# Arquitectura de dos pasadas (soporte de CSV de hasta ~1.7 TiB)
# =============================================================================
# Mismo motivo que la sección equivalente en `src/data_cleaning.py`: separa
# "calcular parámetros globales" (Pasada 1, un recorrido perezoso del CSV
# completo con Dask, reducciones acotadas por columna) de "aplicar esos
# parámetros" (Pasada 2, sobre una única partición ya materializada en
# memoria como `pandas.DataFrame`), de modo que en ningún punto se necesite
# tener el dataset completo cargado a la vez.
_SCALING_COLUMNS: tuple[str, ...] = ("MONTO_APLICADO", "UNIDADES", "PORCENTAJE_DESCUENTO")


@dataclass
class GlobalFeatureStats:
    """
    Parámetros globales calculados en la Pasada 1, suficientes para que
    la Pasada 2 genere las variables derivadas por partición sin volver a
    leer el CSV completo.

    Attributes:
        frecuencia_por_cliente: mapa `CODIGO_CLIENTE -> nº de boletas
            únicas` en TODO el dataset (calculado una sola vez sobre el
            dataset completo vía `groupby().nunique()`, una reducción de
            cardinalidad = nº de clientes, no de transacciones).
        means / stds: parámetros de estandarización por columna, igual
            semántica que `ScalingParameters`.
    """

    frecuencia_por_cliente: dict[str, int] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_clientes": len(self.frecuencia_por_cliente),
            "means": self.means,
            "stds": self.stds,
        }


def compute_global_feature_stats(
    df: dd.DataFrame, include_frecuencia_compra: bool | None = None
) -> GlobalFeatureStats:
    """
    Pasada 1: recorre `df` una vez y calcula los parámetros globales
    necesarios para generar las variables derivadas por partición en la
    Pasada 2.

    `frecuencia_por_cliente` se materializa como un único `dict` de
    Python (cardinalidad = nº de clientes, muchísimo menor que el nº de
    transacciones de un archivo de 1.7 TiB, por lo que cabe cómodamente
    en memoria de un solo proceso) — mismo razonamiento ya documentado en
    `add_frecuencia_compra`.

    Args:
        df: `dask.dataframe.DataFrame` (perezoso).
        include_frecuencia_compra: si se ejecuta el
            `groupby("CODIGO_CLIENTE")` que calcula FRECUENCIA_COMPRA
            (la operación más costosa de esta pasada sobre datasets muy
            grandes, al recorrer todo el archivo agrupando por cliente).
            Si es `None` (por defecto), se usa
            `config.ENABLE_FEATURE_ENGINEERING`.

    Complejidad:
        O(n) tiempo distribuido si se incluye el groupby, O(k) espacio
        adicional (k = número de clientes únicos); O(n) tiempo y O(1)
        espacio adicional si se omite.
    """
    if include_frecuencia_compra is None:
        include_frecuencia_compra = config.ENABLE_FEATURE_ENGINEERING

    result = GlobalFeatureStats()

    if include_frecuencia_compra and "CODIGO_CLIENTE" in df.columns and "BOLETA" in df.columns:
        frecuencia = (
            df.groupby("CODIGO_CLIENTE")["BOLETA"].nunique().compute()
        )
        result.frecuencia_por_cliente = frecuencia.to_dict()
    elif not include_frecuencia_compra:
        logger.info(
            "CPYD_ENABLE_FEATURE_ENGINEERING=false: se omite el groupby "
            "de FRECUENCIA_COMPRA en la Pasada 1."
        )

    for column in _SCALING_COLUMNS:
        if column not in df.columns:
            continue
        series = df[column].astype("float64")
        mean = float(series.mean(skipna=True).compute())
        std = float(series.std(skipna=True, ddof=1).compute())
        result.means[column] = mean
        result.stds[column] = std

    logger.info(
        "Pasada 1 (features) completa: %d cliente(s) únicos, medias=%s, "
        "std=%s.",
        len(result.frecuencia_por_cliente), result.means, result.stds,
    )
    return result


def apply_features_partition(pdf: pd.DataFrame, stats: GlobalFeatureStats) -> pd.DataFrame:
    """
    Pasada 2: genera MONTO_POR_UNIDAD, EDAD, FRECUENCIA_COMPRA y las
    columnas `_STD` sobre UNA partición ya materializada (`pandas.DataFrame`),
    usando únicamente los parámetros de `stats` (ningún estadístico se
    recalcula aquí, ni se vuelve a leer el CSV).

    Args:
        pdf: una partición del dataset, ya como `pandas.DataFrame` (salida
            de `src.data_cleaning.apply_cleaning_partition`).
        stats: salida de `compute_global_feature_stats`.

    Returns:
        pandas.DataFrame: la misma partición, enriquecida.

    Complejidad:
        O(m) tiempo y espacio, m = número de filas de ESTA partición.
    """
    pdf = pdf.copy()

    if {"MONTO_APLICADO", "UNIDADES"}.issubset(pdf.columns):
        safe_units = pdf["UNIDADES"].mask(pdf["UNIDADES"] == 0).astype("float64")
        pdf["MONTO_POR_UNIDAD"] = pdf["MONTO_APLICADO"] / safe_units

    if {"FECHA", "FECHA_NACIMIENTO"}.issubset(pdf.columns):
        years_diff = pdf["FECHA"].dt.year - pdf["FECHA_NACIMIENTO"].dt.year
        birthday_not_yet_occurred = (
            (pdf["FECHA"].dt.month < pdf["FECHA_NACIMIENTO"].dt.month)
            | (
                (pdf["FECHA"].dt.month == pdf["FECHA_NACIMIENTO"].dt.month)
                & (pdf["FECHA"].dt.day < pdf["FECHA_NACIMIENTO"].dt.day)
            )
        )
        edad = (years_diff - birthday_not_yet_occurred.astype("Int64")).astype("float64")
        invalid_age_mask = (edad < 0) | (edad > 110)
        pdf["EDAD"] = edad.mask(invalid_age_mask)

    if "CODIGO_CLIENTE" in pdf.columns and stats.frecuencia_por_cliente:
        # `stats.frecuencia_por_cliente` queda vacío cuando
        # `compute_global_feature_stats` se ejecutó con
        # `include_frecuencia_compra=False` (CPYD_ENABLE_FEATURE_ENGINEERING=false):
        # en ese caso se omite por completo la columna, en vez de crearla
        # llena de NaN.
        pdf["FRECUENCIA_COMPRA"] = (
            pdf["CODIGO_CLIENTE"].map(stats.frecuencia_por_cliente).astype("Int64")
        )

    for column in _SCALING_COLUMNS:
        if column not in pdf.columns or column not in stats.stds:
            continue
        mean = stats.means[column]
        std = stats.stds[column]
        if std == 0 or np.isnan(std):
            continue
        pdf[f"{column}_STD"] = (pdf[column].astype("float64") - mean) / std

    return pdf
