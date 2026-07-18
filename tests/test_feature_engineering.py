"""
tests/test_feature_engineering.py
====================================

Pruebas unitarias para `src/feature_engineering.py`.

Como el módulo ahora opera sobre `dask.dataframe.DataFrame` de punta a
punta (ver docstring de `src/feature_engineering.py`), cada test
construye un `pd.DataFrame` pequeño y lo envuelve con
`dd.from_pandas(..., npartitions=2)` antes de pasarlo a las funciones
bajo prueba, materializando (`.compute()`) el resultado perezoso antes
de cada aserción.
"""

from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_edad,
    add_frecuencia_compra,
    add_monto_por_unidad,
    apply_features_partition,
    compute_global_feature_stats,
    engineer_features,
    standardize_columns,
)


def _to_dask(df: pd.DataFrame, npartitions: int = 2) -> dd.DataFrame:
    """Envuelve un DataFrame pequeño de Pandas como Dask, para los tests."""
    return dd.from_pandas(df, npartitions=npartitions)


def test_add_monto_por_unidad_basic_division() -> None:
    """MONTO_POR_UNIDAD debe ser exactamente MONTO_APLICADO / UNIDADES."""
    df = pd.DataFrame({"MONTO_APLICADO": [1000.0, 2000.0], "UNIDADES": [2, 4]})
    result = add_monto_por_unidad(_to_dask(df)).compute()
    assert list(result["MONTO_POR_UNIDAD"]) == [500.0, 500.0]


def test_add_monto_por_unidad_handles_zero_units() -> None:
    """UNIDADES=0 debe producir NaN, nunca inf ni una excepción."""
    df = pd.DataFrame({"MONTO_APLICADO": [1000.0, 500.0], "UNIDADES": [0, 5]})
    result = add_monto_por_unidad(_to_dask(df)).compute()
    assert np.isnan(result["MONTO_POR_UNIDAD"].iloc[0])
    assert not np.isinf(result["MONTO_POR_UNIDAD"].iloc[0])
    assert result["MONTO_POR_UNIDAD"].iloc[1] == 100.0


def test_add_edad_computes_years_between_dates() -> None:
    """EDAD debe calcularse respecto a FECHA de transacción, no a hoy."""
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2026-01-01"]),
            "FECHA_NACIMIENTO": pd.to_datetime(["1996-01-01"]),
        }
    )
    result = add_edad(_to_dask(df, npartitions=1)).compute()
    assert abs(result["EDAD"].iloc[0] - 30.0) < 0.1


def test_add_edad_flags_invalid_ages_as_nan() -> None:
    """Edades fuera de [0, 110] deben convertirse a NaN, no eliminarse la fila."""
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "FECHA_NACIMIENTO": pd.to_datetime(["2027-01-01", "1900-01-01"]),
        }
    )
    result = add_edad(_to_dask(df)).compute()
    assert len(result) == 2  # ninguna fila eliminada
    assert result["EDAD"].isna().sum() == 2  # ambas inválidas (futura y >110 años)


def test_add_edad_accounts_for_birthday_not_yet_occurred_this_year() -> None:
    """Si el cumpleaños aún no ocurre este año, la edad debe ser un año menor."""
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2026-06-15", "2026-06-15"]),
            # Cumpleaños en diciembre (aún no ocurre) vs. en enero (ya ocurrió).
            "FECHA_NACIMIENTO": pd.to_datetime(["1996-12-25", "1996-01-01"]),
        }
    )
    result = add_edad(_to_dask(df)).compute()
    assert result["EDAD"].iloc[0] == 29.0  # aún no cumple años este 2026
    assert result["EDAD"].iloc[1] == 30.0  # ya cumplió años este 2026


def test_add_edad_does_not_overflow_with_extreme_birth_dates() -> None:
    """
    Regresión: una FECHA_NACIMIENTO extremadamente antigua no debe lanzar
    OverflowError (bug real detectado en el archivo de datos de producción:
    restar datetimes directamente construye un Timedelta limitado a ~292
    años en su representación interna de nanosegundos int64; el cálculo
    por componentes año/mes/día es inmune a esto).
    """
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2026-06-15"]),
            "FECHA_NACIMIENTO": pd.to_datetime(["1700-01-01"]),
        }
    )
    result = add_edad(_to_dask(df, npartitions=1)).compute()  # no debe lanzar excepción
    assert len(result) == 1
    assert result["EDAD"].isna().iloc[0]  # 326 años > 110 -> filtrada como inválida


def test_add_frecuencia_compra_counts_unique_boletas() -> None:
    """FRECUENCIA_COMPRA debe contar boletas ÚNICAS, no filas."""
    df = pd.DataFrame(
        {
            "CODIGO_CLIENTE": ["C1", "C1", "C1", "C2"],
            "BOLETA": [100, 100, 101, 200],  # C1 tiene 2 boletas únicas (100 repetida)
        }
    )
    result = add_frecuencia_compra(_to_dask(df)).compute()
    c1_freq = result.loc[result["CODIGO_CLIENTE"] == "C1", "FRECUENCIA_COMPRA"].iloc[0]
    c2_freq = result.loc[result["CODIGO_CLIENTE"] == "C2", "FRECUENCIA_COMPRA"].iloc[0]
    assert c1_freq == 2
    assert c2_freq == 1


def test_standardize_columns_produces_zero_mean_unit_std() -> None:
    """Una columna estandarizada debe tener media ~0 y std ~1."""
    df = pd.DataFrame({"X": [10.0, 20.0, 30.0, 40.0, 50.0]})
    ddf_result, params = standardize_columns(_to_dask(df), ["X"])
    result = ddf_result.compute()

    assert abs(result["X_STD"].mean()) < 1e-9
    assert abs(result["X_STD"].std(ddof=1) - 1.0) < 1e-9
    assert params.means["X"] == 30.0


def test_standardize_columns_handles_constant_column() -> None:
    """
    Una columna constante (std=0) no debe generar división por cero, y
    NO debe crear una columna `_STD` engañosa (que daría la falsa
    impresión de haber sido estandarizada). Los parámetros (media, std=0)
    igualmente quedan documentados en `ScalingParameters`.
    """
    df = pd.DataFrame({"X": [5.0, 5.0, 5.0]})
    ddf_result, params = standardize_columns(_to_dask(df), ["X"])
    result = ddf_result.compute()

    assert "X_STD" not in result.columns
    assert params.stds["X"] == 0.0
    assert params.means["X"] == 5.0


def test_engineer_features_end_to_end(synthetic_raw_dataframe: pd.DataFrame) -> None:
    """El pipeline completo debe ejecutarse sin excepciones y agregar todas las columnas."""
    from config import COLUMN_RENAME_MAP
    from src.data_cleaning import clean_dataset

    df = synthetic_raw_dataframe.rename(columns=COLUMN_RENAME_MAP)
    ddf_clean, _ = clean_dataset(_to_dask(df))
    ddf_final, scaling_params = engineer_features(ddf_clean)
    df_final = ddf_final.compute()

    for expected_col in ("MONTO_POR_UNIDAD", "EDAD", "FRECUENCIA_COMPRA",
                          "MONTO_APLICADO_STD", "UNIDADES_STD",
                          "PORCENTAJE_DESCUENTO_STD"):
        assert expected_col in df_final.columns

    assert len(df_final) == len(df)  # sin eliminación de filas
    assert "MONTO_APLICADO" in scaling_params.means


# =============================================================================
# Arquitectura de dos pasadas: compute_global_feature_stats / apply_features_partition
# =============================================================================
def test_compute_global_feature_stats_calcula_frecuencia_por_cliente() -> None:
    df = pd.DataFrame(
        {
            "CODIGO_CLIENTE": ["C1", "C1", "C1", "C2"],
            "BOLETA": [100, 100, 101, 200],
        }
    )
    stats = compute_global_feature_stats(_to_dask(df))
    assert stats.frecuencia_por_cliente == {"C1": 2, "C2": 1}


def test_compute_global_feature_stats_calcula_medias_y_std() -> None:
    df = pd.DataFrame({"MONTO_APLICADO": [10.0, 20.0, 30.0, 40.0, 50.0]})
    stats = compute_global_feature_stats(_to_dask(df))
    assert stats.means["MONTO_APLICADO"] == 30.0


def test_apply_features_partition_usa_frecuencia_precalculada() -> None:
    """
    La Pasada 2 no recalcula el groupby: dos particiones distintas deben
    reutilizar el MISMO mapa de frecuencia por cliente calculado en la
    Pasada 1 (incluso si un cliente no aparece completo en una partición).
    """
    df_global = pd.DataFrame(
        {"CODIGO_CLIENTE": ["C1", "C1", "C2"], "BOLETA": [100, 101, 200]}
    )
    stats = compute_global_feature_stats(_to_dask(df_global))

    particion = pd.DataFrame({"CODIGO_CLIENTE": ["C1"], "BOLETA": [100]})
    resultado = apply_features_partition(particion, stats)

    assert resultado["FRECUENCIA_COMPRA"].iloc[0] == 2  # C1 tiene 2 boletas EN TODO el dataset


def test_apply_features_partition_calcula_monto_por_unidad_y_edad() -> None:
    # `df_global` necesita más de una fila para que std(ddof=1) esté bien
    # definida: con una sola muestra la varianza es 0/0 (NaN), lo que
    # genera un RuntimeWarning de división inválida por cada columna de
    # `_SCALING_COLUMNS`. Las dos filas son idénticas a propósito: solo
    # alimentan `stats` (medias/std), mientras que las aserciones del
    # test verifican el resultado de `apply_features_partition` sobre
    # `particion`, más abajo.
    df_global = pd.DataFrame({"MONTO_APLICADO": [1000.0, 1000.0], "UNIDADES": [2, 2]})
    stats = compute_global_feature_stats(_to_dask(df_global))

    particion = pd.DataFrame(
        {
            "MONTO_APLICADO": [1000.0],
            "UNIDADES": [2],
            "FECHA": pd.to_datetime(["2026-01-01"]),
            "FECHA_NACIMIENTO": pd.to_datetime(["1996-01-01"]),
        }
    )
    resultado = apply_features_partition(particion, stats)
    assert resultado["MONTO_POR_UNIDAD"].iloc[0] == 500.0
    assert abs(resultado["EDAD"].iloc[0] - 30.0) < 0.1


def test_apply_features_partition_estandariza_con_media_std_globales() -> None:
    df_global = pd.DataFrame({"MONTO_APLICADO": [10.0, 20.0, 30.0, 40.0, 50.0]})
    stats = compute_global_feature_stats(_to_dask(df_global))

    particion = pd.DataFrame({"MONTO_APLICADO": [30.0]})  # justo la media global
    resultado = apply_features_partition(particion, stats)
    assert abs(resultado["MONTO_APLICADO_STD"].iloc[0]) < 1e-9  # (30-30)/std ≈ 0


# =============================================================================
# Feature engineering opcional (CPYD_ENABLE_FEATURE_ENGINEERING /
# include_frecuencia_compra): FRECUENCIA_COMPRA puede omitirse para
# acelerar la carga en datasets muy grandes.
# =============================================================================
def test_compute_global_feature_stats_omite_frecuencia_si_se_desactiva() -> None:
    df = pd.DataFrame(
        {
            "CODIGO_CLIENTE": ["C1", "C1", "C2"],
            "BOLETA": [100, 101, 200],
            "MONTO_APLICADO": [10.0, 20.0, 30.0],
        }
    )
    stats = compute_global_feature_stats(_to_dask(df), include_frecuencia_compra=False)
    assert stats.frecuencia_por_cliente == {}
    # Las demás estadísticas (medias/std) se siguen calculando igual.
    assert "MONTO_APLICADO" in stats.means


def test_compute_global_feature_stats_respeta_config_por_defecto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import config

    df = pd.DataFrame({"CODIGO_CLIENTE": ["C1", "C1"], "BOLETA": [100, 101]})

    monkeypatch.setattr(config, "ENABLE_FEATURE_ENGINEERING", False)
    stats_desactivado = compute_global_feature_stats(_to_dask(df))
    assert stats_desactivado.frecuencia_por_cliente == {}

    monkeypatch.setattr(config, "ENABLE_FEATURE_ENGINEERING", True)
    stats_activado = compute_global_feature_stats(_to_dask(df))
    assert stats_activado.frecuencia_por_cliente == {"C1": 2}


def test_apply_features_partition_omite_columna_frecuencia_si_no_se_calculo() -> None:
    df_global = pd.DataFrame({"CODIGO_CLIENTE": ["C1", "C1"], "BOLETA": [100, 101]})
    stats = compute_global_feature_stats(_to_dask(df_global), include_frecuencia_compra=False)

    particion = pd.DataFrame({"CODIGO_CLIENTE": ["C1"], "BOLETA": [100]})
    resultado = apply_features_partition(particion, stats)
    assert "FRECUENCIA_COMPRA" not in resultado.columns


def test_engineer_features_omite_frecuencia_compra_si_se_desactiva(
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    from config import COLUMN_RENAME_MAP
    from src.data_cleaning import clean_dataset

    df = synthetic_raw_dataframe.rename(columns=COLUMN_RENAME_MAP)
    ddf_clean, _ = clean_dataset(_to_dask(df))
    ddf_final, _ = engineer_features(ddf_clean, include_frecuencia_compra=False)
    df_final = ddf_final.compute()

    assert "FRECUENCIA_COMPRA" not in df_final.columns
    # Las variables de bajo costo por fila se siguen generando igual.
    assert "MONTO_POR_UNIDAD" in df_final.columns
    assert "EDAD" in df_final.columns
