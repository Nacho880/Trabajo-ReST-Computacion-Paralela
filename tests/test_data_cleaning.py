"""
tests/test_data_cleaning.py
=============================

Pruebas unitarias para `src/data_cleaning.py`.

Como el módulo ahora opera sobre `dask.dataframe.DataFrame` de punta a
punta (ver docstring de `src/data_cleaning.py`), cada test construye un
`pd.DataFrame` pequeño y lo envuelve con `dd.from_pandas(..., npartitions=2)`
antes de pasarlo a las funciones bajo prueba. `npartitions=2` (no 1) es
deliberado: obliga a que las agregaciones (sum, mean, quantile) combinen
resultados de múltiples particiones, ejercitando el mismo camino de
código que corre sobre el dataset real de 10 particiones.

Los resultados perezosos (Series/Scalars de Dask) se materializan
explícitamente con `.compute()` antes de cada aserción.
"""

from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    clean_dataset,
    detect_outliers_iqr,
    detect_outliers_zscore,
    handle_missing_values,
    treat_outliers,
)


def _to_dask(df: pd.DataFrame, npartitions: int = 2) -> dd.DataFrame:
    """Envuelve un DataFrame pequeño de Pandas como Dask, para los tests."""
    return dd.from_pandas(df, npartitions=npartitions)


def test_handle_missing_values_imputes_median() -> None:
    """La imputación de PORCENTAJE_DESCUENTO debe usar la mediana exacta."""
    df = pd.DataFrame(
        {
            "PORCENTAJE_DESCUENTO": [0.1, 0.2, np.nan, 0.4, np.nan],
            "MONTO_APLICADO": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        }
    )
    expected_median = df["PORCENTAJE_DESCUENTO"].median()  # mediana de [0.1,0.2,0.4] = 0.2

    ddf_clean, report = handle_missing_values(_to_dask(df))
    df_clean = ddf_clean.compute()

    assert df_clean["PORCENTAJE_DESCUENTO"].isna().sum() == 0
    # 3 coincidencias: el 0.2 original (índice 1) + los 2 valores imputados.
    assert (df_clean["PORCENTAJE_DESCUENTO"] == expected_median).sum() == 3
    assert report.missing_before["PORCENTAJE_DESCUENTO"] == 2
    assert "imputacion_mediana" in report.missing_treatment["PORCENTAJE_DESCUENTO"]


def test_handle_missing_values_does_not_impute_fecha_nacimiento() -> None:
    """FECHA_NACIMIENTO nula debe conservarse como NaT, nunca imputarse."""
    df = pd.DataFrame(
        {
            "FECHA_NACIMIENTO": pd.to_datetime(["1990-01-01", None, "1985-05-05"]),
            "MONTO_APLICADO": [100.0, 200.0, 300.0],
        }
    )
    ddf_clean, report = handle_missing_values(_to_dask(df))
    df_clean = ddf_clean.compute()

    assert df_clean["FECHA_NACIMIENTO"].isna().sum() == 1
    assert "sin_imputacion" in report.missing_treatment["FECHA_NACIMIENTO"]


def test_detect_outliers_iqr_flags_extreme_value() -> None:
    """Un valor extremo (100x la escala normal) debe detectarse por IQR."""
    values = [10, 11, 9, 10, 12, 11, 10, 1000]  # 1000 es outlier evidente
    df = pd.DataFrame({"MONTO_APLICADO": values})

    mask = detect_outliers_iqr(_to_dask(df), "MONTO_APLICADO").compute()

    assert mask.iloc[-1] == True  # noqa: E712 - claridad de intención en test
    assert mask.iloc[:-1].sum() == 0


def test_detect_outliers_zscore_flags_extreme_value() -> None:
    """Un valor extremo también debe detectarse por Z-score."""
    values = [10, 11, 9, 10, 12, 11, 10, 1000]
    df = pd.DataFrame({"MONTO_APLICADO": values})

    mask = detect_outliers_zscore(_to_dask(df), "MONTO_APLICADO", threshold=2.0).compute()

    assert mask.iloc[-1] == True  # noqa: E712


def test_detect_outliers_zscore_constant_column_returns_no_outliers() -> None:
    """Una columna constante (std=0) no debe generar división por cero ni outliers."""
    df = pd.DataFrame({"X": [5, 5, 5, 5]})
    mask = detect_outliers_zscore(_to_dask(df), "X").compute()
    assert mask.sum() == 0


def test_treat_outliers_does_not_remove_rows() -> None:
    """La política del proyecto es marcar outliers, nunca eliminar filas."""
    values = [10, 11, 9, 10, 12, 11, 10, 1000]
    df = pd.DataFrame({"MONTO_APLICADO": values})

    ddf_out, n_outliers = treat_outliers(_to_dask(df), "MONTO_APLICADO", method="iqr")
    df_out = ddf_out.compute()

    assert len(df_out) == len(df)  # ninguna fila eliminada
    assert n_outliers == 1
    assert "MONTO_APLICADO_ES_OUTLIER" in df_out.columns
    assert df_out["MONTO_APLICADO_ES_OUTLIER"].sum() == 1


def test_treat_outliers_invalid_method_raises() -> None:
    """Un método desconocido debe lanzar ValueError, no fallar silenciosamente."""
    df = pd.DataFrame({"X": [1, 2, 3]})
    try:
        treat_outliers(_to_dask(df), "X", method="metodo_inexistente")
        assert False, "Debió lanzar ValueError"
    except ValueError:
        pass


def test_clean_dataset_end_to_end(synthetic_raw_dataframe: pd.DataFrame) -> None:
    """
    Prueba de integración: el pipeline completo de limpieza debe
    ejecutarse sin excepciones sobre el dataset sintético (envuelto en
    Dask) y no debe eliminar ninguna fila (política de marcado, no
    eliminación).
    """
    from config import COLUMN_RENAME_MAP

    df = synthetic_raw_dataframe.rename(columns=COLUMN_RENAME_MAP)
    ddf_clean, report = clean_dataset(_to_dask(df))
    df_clean = ddf_clean.compute()

    assert len(df_clean) == len(df)  # sin eliminación de filas
    assert df_clean["PORCENTAJE_DESCUENTO"].isna().sum() == 0
    assert "MONTO_APLICADO" in report.outliers_detected
    assert report.outliers_detected["MONTO_APLICADO"] >= 1  # el outlier inyectado


# =============================================================================
# Arquitectura de dos pasadas: compute_global_cleaning_stats / apply_cleaning_partition
# =============================================================================
from src.data_cleaning import GlobalCleaningStats, apply_cleaning_partition, compute_global_cleaning_stats  # noqa: E402


def test_compute_global_cleaning_stats_calcula_mediana_exacta() -> None:
    df = pd.DataFrame(
        {
            "PORCENTAJE_DESCUENTO": [0.1, 0.2, np.nan, 0.4, np.nan],
            "MONTO_APLICADO": [10, 11, 9, 10, 1000.0],
            "UNIDADES": [1, 1, 1, 1, 1],
        }
    )
    stats = compute_global_cleaning_stats(_to_dask(df))

    assert stats.median_by_column["PORCENTAJE_DESCUENTO"] == pytest.approx(0.2)
    assert stats.missing_before["PORCENTAJE_DESCUENTO"] == 2
    assert "MONTO_APLICADO" in stats.outlier_bounds


def test_apply_cleaning_partition_usa_stats_precalculadas_sin_recalcular() -> None:
    """
    La Pasada 2 no debe recalcular nada global: dos particiones distintas
    con la MISMA `stats` deben imputarse con el MISMO valor de mediana.
    """
    df_global = pd.DataFrame({"PORCENTAJE_DESCUENTO": [0.1, 0.2, 0.3, np.nan]})
    stats = compute_global_cleaning_stats(_to_dask(df_global))

    particion_a = pd.DataFrame({"PORCENTAJE_DESCUENTO": [np.nan, 0.5]})
    particion_b = pd.DataFrame({"PORCENTAJE_DESCUENTO": [np.nan]})

    resultado_a = apply_cleaning_partition(particion_a, stats)
    resultado_b = apply_cleaning_partition(particion_b, stats)

    assert resultado_a["PORCENTAJE_DESCUENTO"].iloc[0] == stats.median_by_column["PORCENTAJE_DESCUENTO"]
    assert resultado_b["PORCENTAJE_DESCUENTO"].iloc[0] == stats.median_by_column["PORCENTAJE_DESCUENTO"]


def test_apply_cleaning_partition_marca_outliers_sin_eliminar_filas() -> None:
    df_global = pd.DataFrame({"MONTO_APLICADO": [10, 11, 9, 10, 12, 11, 10, 1000.0]})
    stats = compute_global_cleaning_stats(_to_dask(df_global))

    particion = pd.DataFrame({"MONTO_APLICADO": [10.0, 1000.0]})
    resultado = apply_cleaning_partition(particion, stats)

    assert len(resultado) == len(particion)
    assert resultado["MONTO_APLICADO_ES_OUTLIER"].tolist() == [False, True]


def test_apply_cleaning_partition_sobre_particion_vacia_no_falla() -> None:
    stats = GlobalCleaningStats(median_by_column={"PORCENTAJE_DESCUENTO": 0.2})
    particion_vacia = pd.DataFrame({"PORCENTAJE_DESCUENTO": []})
    resultado = apply_cleaning_partition(particion_vacia, stats)
    assert resultado.empty
