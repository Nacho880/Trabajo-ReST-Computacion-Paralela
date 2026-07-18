"""
src/data_cleaning.py
=====================

Tratamiento de valores faltantes y detección de outliers.

Responsabilidad única: recibir un `dask.dataframe.DataFrame` (el
resultado de `data_loader.load_and_validate(...)`, ya persistido en
memoria vía `.persist()`, pero aún como colección Dask
de cara a las transformaciones) y devolver una versión limpia,
también como `dask.dataframe.DataFrame`, documentando en el log cada
decisión tomada (cuántos nulos había, qué método de imputación se
usó, cuántos outliers se detectaron y cómo se trataron).

Diseño: Dask de punta a punta en esta etapa, no Pandas.
    Este módulo está escrito directamente contra la API de
    `dask.dataframe`, que es un espejo deliberado de la de Pandas.
    Las transformaciones sobre las filas del dataset (imputación,
    marcado de outliers) se mantienen perezosas -Dask construye el
    grafo de tareas sin ejecutar nada nuevo- y solo se materializan
    (`.compute()`) los pocos VALORES ESCALARES PEQUEÑOS que son
    estrictamente necesarios:
        1. Para una decisión de control de flujo en Python (ej.
           "if std == 0"), que no puede evaluarse sobre un objeto
           perezoso.
        2. Para pasar datos a `scipy.stats`, que no entiende objetos
           Dask y requiere arreglos de NumPy concretos.
        3. Para dejar constancia en el log/reporte de un número
           concreto (ej. "se imputó con mediana=X").

    Rendimiento: quien orquesta la carga (`app/services/dataset.py`)
    llama a `.persist()` sobre el DataFrame crudo inmediatamente después
    de `load_and_validate(...)`, ANTES de invocar `clean_dataset`. Esto es lo que hace viable tener múltiples
    `.compute()` de valores pequeños en este módulo: cada uno opera
    sobre particiones ya materializadas en memoria, no repite la
    lectura/parseo completo del CSV desde cero (que sí ocurriría si el
    DataFrame siguiera perezoso desde la lectura).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """
    Reporte estructurado de las decisiones tomadas durante la limpieza.

    Se retorna junto al DataFrame limpio para que pueda documentarse en
    el informe técnico sin tener que parsear logs de texto plano. Todos
    los valores almacenados aquí son ESCALARES CONCRETOS (ya
    materializados con `.compute()` en el momento en que se calcularon),
    nunca objetos perezosos de Dask.

    Attributes:
        missing_before: conteo de nulos por columna, antes de imputar.
        missing_treatment: método aplicado por columna
            ("mediana", "moda", "eliminacion_fila", etc.).
        mcar_test_pvalue: p-value del test de aleatoriedad de los nulos
            (aproximación de MCAR vía comparación de medias con/sin nulo,
            ver `_mcar_like_test`), o None si no aplica.
        outliers_detected: conteo de outliers detectados por columna.
        outliers_treatment: descripción de cómo se trató cada columna
            con outliers ("marcado, no eliminado" es la política por
            defecto: ver docstring de `detect_outliers_iqr`).
        rows_before: número de filas antes de la limpieza.
        rows_after: número de filas después de la limpieza.
    """

    missing_before: dict[str, int] = field(default_factory=dict)
    missing_treatment: dict[str, str] = field(default_factory=dict)
    mcar_test_pvalue: dict[str, float] = field(default_factory=dict)
    outliers_detected: dict[str, int] = field(default_factory=dict)
    outliers_treatment: dict[str, str] = field(default_factory=dict)
    rows_before: int = 0
    rows_after: int = 0

    def to_dict(self) -> dict:
        """Serializa el reporte a un diccionario plano apto para JSON."""
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "missing_before": self.missing_before,
            "missing_treatment": self.missing_treatment,
            "mcar_test_pvalue": self.mcar_test_pvalue,
            "outliers_detected": self.outliers_detected,
            "outliers_treatment": self.outliers_treatment,
        }


def _mcar_like_test(
    df: dd.DataFrame, target_column: str, probe_column: str
) -> float | None:
    """
    Aproxima un test de tipo MCAR (Missing Completely At Random) para
    `target_column`, comparando la distribución de `probe_column` entre
    el grupo con nulo en `target_column` y el grupo sin nulo, mediante
    Mann-Whitney U (no paramétrico, no asume normalidad).

    Racional estadístico:
        Si los datos son MCAR, la ausencia de `target_column` no debería
        estar relacionada con los valores observados de ninguna otra
        variable. Si `probe_column` difiere significativamente entre
        ambos grupos, hay evidencia en contra de MCAR (podría ser MAR:
        Missing At Random, dependiente de variables observadas).

    Materialización necesaria (única excepción a "todo perezoso"):
        `scipy.stats.mannwhitneyu` requiere arreglos de NumPy concretos,
        no objetos Dask. Se materializan aquí SOLO los dos grupos
        comparados (con nulo / sin nulo en `target_column`), un
        subconjunto acotado de una sola columna -no el dataset
        completo-, vía `.compute()` explícito.

    Args:
        df: `dask.dataframe.DataFrame` con los datos originales
            (incluyendo nulos).
        target_column: columna cuyo patrón de nulos se evalúa.
        probe_column: columna numérica usada como variable de sondeo.

    Returns:
        float | None: p-value del test. None si no hay suficientes
        datos en algún grupo (mínimo 2 observaciones por grupo).

    Complejidad:
        O(n) tiempo distribuido para identificar los grupos, O(m) para
        el test en sí, m = tamaño de los grupos materializados.
    """
    if target_column not in df.columns or probe_column not in df.columns:
        return None

    is_missing = df[target_column].isna()
    group_missing = df.loc[is_missing, probe_column].dropna().compute()
    group_present = df.loc[~is_missing, probe_column].dropna().compute()

    if len(group_missing) < 2 or len(group_present) < 2:
        return None

    _, p_value = stats.mannwhitneyu(
        group_missing, group_present, alternative="two-sided"
    )
    return float(p_value)


# Columnas opcionales tratadas por `handle_missing_values`. Se declaran
# como configuración explícita (no como literales repetidos dentro de la
# función) para que agregar/quitar una columna opcional en el futuro no
# requiera duplicar el bloque de lógica, y para que la ausencia de
# CUALQUIERA de ellas en `df` (columna que no vino en un subconjunto de
# datos, un test, o una versión futura del CSV) sea un caso normal y no
# una excepción.
_MEDIAN_IMPUTE_COLUMNS: tuple[str, ...] = ("PORCENTAJE_DESCUENTO",)
_MCAR_PROBE_COLUMN = "MONTO_APLICADO"
_NEVER_IMPUTE_COLUMNS: tuple[str, ...] = ("FECHA_NACIMIENTO",)


def _exact_quantile(series: dd.Series, q: float) -> float:
    """
    Cuantil exacto de una `dask.dataframe.Series`, materializando SOLO
    esa columna (no el DataFrame completo) y delegando en
    `pandas.Series.quantile`, que es exacto por definición.

    Mismo motivo que en `detect_outliers_iqr` (ver su "Nota de diseño"):
    `dask.dataframe.Series.quantile(...)` usa por defecto un algoritmo
    APROXIMADO, que puede desviarse del valor real con pocas particiones
    o pocos datos -exactamente el escenario de los tests unitarios, que
    usan `npartitions=2` sobre 3-5 filas-. Para una decisión reproducible
    y auditable como la imputación de faltantes, se requiere el valor
    exacto.

    Args:
        series: columna (perezosa) de la que se calcula el cuantil.
        q: cuantil deseado, en [0, 1].

    Returns:
        float: valor exacto del cuantil `q`.

    Complejidad:
        O(n) para materializar la columna, O(n log n) para el cuantil
        exacto sobre ella (una sola columna, no el dataset completo).
    """
    return float(series.compute().quantile(q))


def handle_missing_values(
    df: dd.DataFrame, alpha: float = 0.05
) -> tuple[dd.DataFrame, CleaningReport]:
    """
    Identifica y trata valores faltantes en las columnas opcionales del
    dataset: las declaradas en `_MEDIAN_IMPUTE_COLUMNS` (numéricas, ej.
    `PORCENTAJE_DESCUENTO`) y en `_NEVER_IMPUTE_COLUMNS` (ej.
    `FECHA_NACIMIENTO`). Opera de forma perezosa sobre un
    `dask.dataframe.DataFrame`.

    Robustez ante columnas opcionales ausentes:
        Ninguna columna de `_MEDIAN_IMPUTE_COLUMNS` / `_NEVER_IMPUTE_COLUMNS`
        es obligatoria en `df`. Si una columna no está presente (un
        subconjunto de datos, un test unitario con un esquema reducido, o
        una versión futura del CSV que la omita), esa columna simplemente
        se salta sin lanzar excepción: ni `handle_missing_values` ni
        `clean_dataset` asumen un esquema fijo de columnas más allá de lo
        que efectivamente venga en `df`.

    Estrategia:
        - Columnas en `_MEDIAN_IMPUTE_COLUMNS`: se imputan con la MEDIANA
          EXACTA (`_exact_quantile`; robusta a outliers, apropiada para
          una variable acotada en [0, 1] con posible asimetría). Si
          `_MCAR_PROBE_COLUMN` también está presente, se contrasta con el
          test tipo MCAR (`_mcar_like_test`), documentando el resultado
          en el reporte independientemente de la conclusión.
        - Columnas en `_NEVER_IMPUTE_COLUMNS`: no se imputan (ej.
          imputar una fecha de nacimiento inventaría una edad falsa). Se
          conservan como NaT/NaN y se excluyen fila a fila SOLO en los
          análisis que específicamente las requieran.

    Args:
        df: `dask.dataframe.DataFrame` crudo (recién cargado y renombrado).
        alpha: nivel de significancia para el test de tipo MCAR.

    Returns:
        tuple[dd.DataFrame, CleaningReport]: DataFrame perezoso con las
        columnas imputables tratadas, y el reporte de las decisiones
        tomadas (con valores ya concretos).

    Complejidad:
        O(n) tiempo distribuido entre particiones de Dask para las
        reducciones (conteo de nulos, mediana); las transformaciones de
        columnas permanecen perezosas (costo O(1) de encolado).
    """
    report = CleaningReport(rows_before=len(df))

    numeric_null_counts = df.isna().sum().compute()
    report.missing_before = {
        col: int(count) for col, count in numeric_null_counts.items() if count > 0
    }
    logger.info("Valores faltantes detectados por columna: %s", report.missing_before)

    # --- Columnas numéricas: imputación por mediana exacta ---
    # IMPORTANTE: se verifica primero la existencia de la columna. Acceder
    # a df[col] antes de confirmar que la columna existe lanza KeyError
    # incluso si esa comprobación aparece luego en un `and`, porque el
    # acceso ocurre en una sentencia previa, no de forma perezosa dentro
    # de la condición. Por eso cada columna opcional se trata en su
    # propio bloque `if col in df.columns`, nunca fuera de él.
    for column in _MEDIAN_IMPUTE_COLUMNS:
        if column not in df.columns:
            continue

        has_missing = bool(df[column].isna().any().compute())
        if not has_missing:
            continue

        if _MCAR_PROBE_COLUMN in df.columns:
            p_value = _mcar_like_test(df, column, _MCAR_PROBE_COLUMN)
            if p_value is not None:
                report.mcar_test_pvalue[column] = p_value
                conclusion = (
                    "no se rechaza MCAR (p>=alpha)" if p_value >= alpha
                    else "se rechaza MCAR, posible MAR (p<alpha)"
                )
                logger.info(
                    "Test tipo MCAR para %s vs %s: p-value=%.4f -> %s",
                    column, _MCAR_PROBE_COLUMN, p_value, conclusion,
                )

        median_value = _exact_quantile(df[column], 0.5)
        df[column] = df[column].fillna(median_value)
        report.missing_treatment[column] = f"imputacion_mediana(valor={median_value:.4f})"
        logger.info(
            "%s: %d nulos imputados con mediana=%.4f",
            column, report.missing_before.get(column, 0), median_value,
        )

    # --- Columnas que se conservan tal cual (NaT/NaN), nunca se imputan ---
    for column in _NEVER_IMPUTE_COLUMNS:
        if column not in df.columns:
            continue

        has_missing = bool(df[column].isna().any().compute())
        if not has_missing:
            continue

        report.missing_treatment[column] = (
            "sin_imputacion(se_excluye_fila_a_fila_solo_en_analisis_que_la_requieran)"
        )
        logger.info(
            "%s: %d nulos NO se imputan (evita inventar valores); se "
            "excluirán fila a fila solo en cálculos que dependan de esta "
            "columna.",
            column, report.missing_before.get(column, 0),
        )

    report.rows_after = len(df)
    return df, report


def detect_outliers_iqr(df: dd.DataFrame, column: str, k: float = 1.5) -> dd.Series:
    """
    Detecta outliers univariados usando el método del Rango
    Intercuartílico (IQR). El marcado fila a fila se mantiene perezoso
    (Dask): solo se materializan los dos cuantiles (Q1, Q3) como
    escalares concretos de Python -exactos, no aproximados, ver "Nota
    de diseño" más abajo-, y la máscara resultante se construye
    comparando la Series perezosa de Dask contra esos dos escalares.

    Un valor se marca como outlier si está fuera de
    [Q1 - k*IQR, Q3 + k*IQR], con k=1.5 (convención estándar de Tukey).

    Args:
        df: `dask.dataframe.DataFrame` con la columna a analizar.
        column: nombre de la columna numérica.
        k: multiplicador del IQR.

    Returns:
        dd.Series[bool]: máscara booleana perezosa, True donde el valor
        es outlier.

    Complejidad:
        O(n) tiempo distribuido para materializar la columna y O(n log n)
        para el cálculo exacto de cuantiles sobre ella (una sola vez,
        una sola columna), O(1) espacio adicional para la máscara, que
        se mantiene perezosa (comparación contra dos escalares concretos
        sobre la Series de Dask original).

    Nota de diseño (cuantiles exactos, no aproximados):
        `dask.dataframe.Series.quantile(...)` usa por defecto un
        algoritmo APROXIMADO (fusiona cuantiles parciales calculados por
        partición). Esa aproximación es aceptable para exploración
        rápida en datasets enormes, pero se vuelve imprecisa -y quiebra
        la garantía estadística del IQR- cuando hay pocas particiones y/o
        valores extremos concentrados en una sola partición (el cuantil
        estimado se desvía del real, marcando como outliers valores que
        no lo son, o viceversa). Como la detección de outliers es una
        decisión de negocio que debe ser reproducible y auditable, se
        calculan los cuantiles EXACTOS materializando únicamente la
        columna en cuestión (`.compute()`, una sola Series, no el
        DataFrame completo) y usando `pandas.Series.quantile`. El
        resultado (dos escalares de Python) se compara luego contra la
        Series perezosa de Dask, de modo que el marcado fila a fila
        sigue siendo una operación distribuida/perezosa como en el
        resto del pipeline.
    """
    # Cast a float64 (dtype nativo de NumPy) antes de cuantiles: columnas
    # con dtype nullable de Pandas (ej. "Int64", usada en UNIDADES) no son
    # interpretables por `dask.array.percentile`, que internamente llama a
    # `np.issubdtype(a.dtype, ...)` y no reconoce `Int64Dtype()` como un
    # dtype válido (lanza TypeError). float64 preserva la semántica de
    # nulos (pd.NA -> NaN) sin alterar el dtype original de `df[column]`.
    series = df[column].astype("float64")

    # Cuantiles exactos: se materializa SOLO esta columna (no el resto del
    # DataFrame) y se usa pandas.Series.quantile, que es exacto por
    # definición. Mismo patrón ya usado en detect_outliers_zscore (mean/std
    # concretos) y en handle_missing_values (mediana concreta).
    concrete_series = series.compute()
    q1 = float(concrete_series.quantile(0.25))
    q3 = float(concrete_series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(
    df: dd.DataFrame, column: str, threshold: float = 3.0
) -> dd.Series:
    """
    Detecta outliers univariados usando el Z-score (desviaciones
    estándar respecto a la media).

    Materialización necesaria: std debe ser un valor concreto de
    Python para la decisión de control de flujo (if std == 0).
    mean se materializa junto a std por simplicidad.

    Args:
        df: `dask.dataframe.DataFrame` con la columna a analizar.
        column: nombre de la columna numérica.
        threshold: desviaciones estándar a partir de las cuales un
            valor se considera outlier.

    Returns:
        dd.Series[bool]: máscara booleana (perezosa en el caso general).

    Complejidad:
        O(n) tiempo distribuido para mean/std, O(1) espacio mientras el
        resto de la máscara se mantenga perezoso.
    """
    # Mismo motivo que en detect_outliers_iqr: se castea a float64 para
    # evitar incompatibilidades de dtypes nullable de Pandas (ej. "Int64")
    # con las operaciones internas de Dask.
    series = df[column].astype("float64")
    mean = float(series.mean().compute())
    std = float(series.std(ddof=1).compute())
    if std == 0 or np.isnan(std):
        # Columna constante: no hay outliers por definición de dispersión.
        return series.isna() & False
    z_scores = (series - mean) / std
    return z_scores.abs() > threshold


def treat_outliers(
    df: dd.DataFrame, column: str, method: str = "iqr"
) -> tuple[dd.DataFrame, int]:
    """
    Detecta outliers en `column` usando el método indicado y los marca
    en una nueva columna booleana `<column>_ES_OUTLIER`, SIN eliminarlos
    del dataset. La asignación de la columna de marca permanece
    perezosa; solo el conteo total (para el log) se materializa.

    Decisión de diseño (justificada para el informe):
        No se eliminan filas con outliers automáticamente. En un
        contexto de ventas de farmacia, un monto inusualmente alto
        puede ser un caso de negocio legítimo (compra al por mayor,
        producto de alto valor) y no necesariamente un error de
        registro. Se prefiere marcar y dejar la decisión de exclusión
        a los análisis específicos que lo requieran, documentando
        explícitamente esta política.

    Args:
        df: `dask.dataframe.DataFrame` a procesar.
        column: columna numérica sobre la que detectar outliers.
        method: "iqr" o "zscore".

    Returns:
        tuple[dd.DataFrame, int]: DataFrame perezoso con la columna de
        marca agregada, y el conteo total de outliers (ya materializado).

    Raises:
        ValueError: si `method` no es "iqr" ni "zscore".

    Complejidad:
        O(n log n) para "iqr", O(n) para "zscore", distribuido entre
        particiones de Dask.
    """
    if method == "iqr":
        mask = detect_outliers_iqr(df, column)
    elif method == "zscore":
        mask = detect_outliers_zscore(df, column)
    else:
        raise ValueError(f"Método de detección de outliers desconocido: '{method}'.")

    flag_column = f"{column}_ES_OUTLIER"
    df[flag_column] = mask
    n_outliers = int(mask.sum().compute())
    n_total = len(df)

    logger.info(
        "Detección de outliers en '%s' (método=%s): %d de %d filas marcadas "
        "(%.2f%%). No se eliminan; se marcan en '%s'.",
        column, method, n_outliers, n_total, 100 * n_outliers / max(n_total, 1),
        flag_column,
    )

    return df, n_outliers


def clean_dataset(df: dd.DataFrame) -> tuple[dd.DataFrame, CleaningReport]:
    """
    Orquesta el pipeline completo de limpieza, de forma perezosa:
        1. Tratamiento de valores faltantes.
        2. Detección de outliers en `MONTO_APLICADO` y `UNIDADES`
           (marcado, no eliminación; ver `treat_outliers`).

    El `dask.dataframe.DataFrame` retornado NO ha sido materializado a
    Pandas: todas las transformaciones están encoladas en el grafo de
    tareas de Dask (aunque el DataFrame de entrada ya esté persistido
    en memoria). La materialización final a Pandas
    (`.compute()`) ocurre después de que también
    `feature_engineering.engineer_features()` haya terminado de
    encolar sus propias transformaciones.

    Args:
        df: `dask.dataframe.DataFrame` crudo, ya renombrado por
            `data_loader` y persistido en memoria.

    Returns:
        tuple[dd.DataFrame, CleaningReport]: dataset limpio (perezoso)
        y reporte consolidado (con valores ya concretos).

    Complejidad:
        O(n log n) tiempo distribuido (dominado por la detección de
        outliers IQR), O(1) espacio adicional mientras se mantenga
        perezoso.
    """
    df_clean, report = handle_missing_values(df)

    for column in ("MONTO_APLICADO", "UNIDADES"):
        if column in df_clean.columns:
            df_clean, n_outliers = treat_outliers(df_clean, column, method="iqr")
            report.outliers_detected[column] = n_outliers
            report.outliers_treatment[column] = (
                "marcado_en_columna_ES_OUTLIER_sin_eliminacion"
            )

    logger.info(
        "Limpieza (Dask, perezosa) encolada. Filas: %d -> %d (sin eliminación de "
        "filas; el grafo de tareas aún no se ha ejecutado por completo).",
        report.rows_before, len(df_clean),
    )
    return df_clean, report


# =============================================================================
# Arquitectura de dos pasadas (soporte de CSV de hasta ~1.7 TiB)
# =============================================================================
# Las funciones de arriba (`clean_dataset` y las que orquesta) procesan un
# `dask.dataframe.DataFrame` de punta a punta y suponen que, en algún punto
# aguas abajo, se puede pagar un `.compute()` sobre el DataFrame COMPLETO.
# Eso es viable mientras el CSV quepa en memoria, pero deja de serlo con un
# archivo de ~1.7 TiB.
#
# Las funciones de esta sección resuelven exactamente el mismo problema
# (imputación de nulos + marcado de outliers) pero separadas en dos etapas
# explícitas, para poder procesar el CSV en dos pasadas sin materializar el
# dataset completo en memoria en ningún momento:
#
#   Pasada 1 — `compute_global_cleaning_stats(ddf)`: recorre el CSV una vez
#   (perezoso, vía Dask) y calcula ÚNICAMENTE escalares globales (mediana de
#   imputación, cotas de outliers). Cada `.compute()` interno opera sobre
#   una sola columna/reducción, nunca sobre el DataFrame completo — mismo
#   patrón ya usado en `handle_missing_values`/`detect_outliers_iqr`.
#
#   Pasada 2 — `apply_cleaning_partition(pdf, stats)`: recibe una ÚNICA
#   partición ya materializada como `pandas.DataFrame` (una fracción
#   pequeña del CSV, entregada por `ddf.to_delayed()` en
#   `app/services/dataset.py`) y le aplica las decisiones de la Pasada 1.
#   No vuelve a leer el CSV ni recalcula ningún estadístico global: solo
#   aplica los valores ya calculados.
def _iqr_bounds_from_series(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


@dataclass
class GlobalCleaningStats:
    """
    Estadísticos globales calculados en la Pasada 1, suficientes para que
    la Pasada 2 impute/marque cada partición sin volver a leer el CSV
    completo.

    Attributes:
        median_by_column: mediana exacta usada para imputar cada columna
            de `_MEDIAN_IMPUTE_COLUMNS` presente en el dataset.
        outlier_bounds: `(cota_inferior, cota_superior)` IQR por columna
            de `_OUTLIER_COLUMNS` presente en el dataset.
        missing_before: conteo global de nulos por columna (informativo,
            para el reporte).
        rows_before: número total de filas del dataset (informativo).
    """

    median_by_column: dict[str, float] = field(default_factory=dict)
    outlier_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    missing_before: dict[str, int] = field(default_factory=dict)
    rows_before: int = 0

    def to_dict(self) -> dict:
        return {
            "rows_before": self.rows_before,
            "missing_before": self.missing_before,
            "median_by_column": self.median_by_column,
            "outlier_bounds": {k: list(v) for k, v in self.outlier_bounds.items()},
        }


_OUTLIER_COLUMNS: tuple[str, ...] = ("MONTO_APLICADO", "UNIDADES")


def compute_global_cleaning_stats(ddf: dd.DataFrame) -> GlobalCleaningStats:
    """
    Pasada 1: recorre `ddf` una vez y calcula los estadísticos globales
    necesarios para limpiar el dataset por partición en la Pasada 2.

    Reutiliza `_exact_quantile` para la mediana de imputación y el mismo
    cálculo de cuantiles exactos que `detect_outliers_iqr` para las cotas
    de outliers — la única diferencia es que aquí el resultado se
    devuelve como datos (`GlobalCleaningStats`) en vez de aplicarse
    inmediatamente sobre `ddf`.

    Complejidad:
        O(n) tiempo distribuido por cada columna analizada (cada una es
        una reducción independiente sobre una sola columna, nunca sobre
        el DataFrame completo); O(1) espacio adicional.
    """
    stats_result = GlobalCleaningStats(rows_before=len(ddf))

    null_counts = ddf.isna().sum().compute()
    stats_result.missing_before = {
        col: int(count) for col, count in null_counts.items() if count > 0
    }

    for column in _MEDIAN_IMPUTE_COLUMNS:
        if column in ddf.columns:
            stats_result.median_by_column[column] = _exact_quantile(ddf[column], 0.5)

    for column in _OUTLIER_COLUMNS:
        if column in ddf.columns:
            series = ddf[column].astype("float64").compute()
            stats_result.outlier_bounds[column] = _iqr_bounds_from_series(series)

    logger.info(
        "Pasada 1 (limpieza) completa: %d fila(s) totales, medianas=%s, "
        "cotas de outliers=%s.",
        stats_result.rows_before, stats_result.median_by_column,
        stats_result.outlier_bounds,
    )
    return stats_result


def apply_cleaning_partition(pdf: pd.DataFrame, stats: GlobalCleaningStats) -> pd.DataFrame:
    """
    Pasada 2: aplica las decisiones de `stats` sobre UNA partición ya
    materializada (`pandas.DataFrame`), sin leer el CSV nuevamente ni
    recalcular ningún estadístico global.

    - Imputa `_MEDIAN_IMPUTE_COLUMNS` con `stats.median_by_column`.
    - Marca (no elimina) outliers en `_OUTLIER_COLUMNS` según
      `stats.outlier_bounds`, en una columna `<col>_ES_OUTLIER`.
    - `_NEVER_IMPUTE_COLUMNS` se dejan tal cual (misma política que
      `handle_missing_values`).

    Args:
        pdf: una partición del dataset, ya como `pandas.DataFrame`.
        stats: salida de `compute_global_cleaning_stats`.

    Returns:
        pandas.DataFrame: la misma partición, limpia (mismo número de
        filas: política de marcado, no de eliminación).

    Complejidad:
        O(m) tiempo y espacio, m = número de filas de ESTA partición
        únicamente.
    """
    pdf = pdf.copy()

    for column, median_value in stats.median_by_column.items():
        if column in pdf.columns:
            pdf[column] = pdf[column].fillna(median_value)

    for column, (lower, upper) in stats.outlier_bounds.items():
        if column not in pdf.columns:
            continue
        serie = pdf[column].astype("float64")
        pdf[f"{column}_ES_OUTLIER"] = (serie < lower) | (serie > upper)

    return pdf
