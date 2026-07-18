"""
tests/conftest.py
==================

Fixtures compartidas por toda la suite de tests.

Se genera un dataset sintético pequeño (no el CSV real de producción)
para que los tests sean rápidos, deterministas y no dependan de la
descarga del archivo real desde Google Drive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Semilla fija, propia de los tests, para que el dataset sintético sea
# determinista (independiente de cualquier configuración externa).
SEED = 42


@pytest.fixture(scope="session")
def synthetic_raw_dataframe() -> pd.DataFrame:
    """
    Construye un DataFrame sintético de 200 filas con la MISMA
    estructura (nombres de columna crudos, con espacios/tildes) que el
    CSV real descrito en el enunciado del proyecto.

    Incluye deliberadamente:
        - Valores nulos en PORCENTAJE DESCUENTO y FECHA NACIMIENTO.
        - Un outlier extremo en MONTO APLICADO.
        - Dos canales (POS, WEB, APP) y tres locales para permitir
          pruebas de Chi-cuadrado, ANOVA y t-test.

    Returns:
        pd.DataFrame: dataset sintético con columnas crudas (sin renombrar).
    """
    rng = np.random.default_rng(SEED)
    n = 200

    canales = rng.choice(["POS", "WEB", "APP"], size=n, p=[0.5, 0.25, 0.25])
    locales = rng.choice([1001, 1002, 1003], size=n)
    unidades = rng.integers(1, 5, size=n)
    descuento = rng.uniform(0.0, 0.3, size=n)
    # Introducimos nulos deliberados en el descuento (~10%).
    null_idx = rng.choice(n, size=int(n * 0.1), replace=False)
    descuento[null_idx] = np.nan

    precio_base = rng.uniform(1000, 20000, size=n)
    monto = precio_base * unidades * (1 - np.nan_to_num(descuento))
    # Outlier extremo intencional.
    monto[0] = monto.max() * 50

    fechas = pd.date_range("2026-01-01", periods=n, freq="h")

    fecha_nac = pd.to_datetime(
        rng.integers(
            pd.Timestamp("1950-01-01").value,
            pd.Timestamp("2005-01-01").value,
            size=n,
        )
    )
    fecha_nac = fecha_nac.to_series().reset_index(drop=True)
    fecha_nac_null_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    fecha_nac = fecha_nac.astype("datetime64[ns]")
    fecha_nac.iloc[fecha_nac_null_idx] = pd.NaT

    codigo_cliente = [f"CLI-{i % 50:04d}" for i in range(n)]  # 50 clientes distintos

    df = pd.DataFrame(
        {
            "FECHA": fechas,
            "CANAL": canales,
            "SKU": rng.integers(1000, 2000, size=n),
            "PRODUCTO": [f"PRODUCTO_{i % 20}" for i in range(n)],
            "UNIDADES": unidades,
            "PORCENTAJE DESCUENTO": descuento,
            "MONTO APLICADO": monto,
            "BOLETA": np.arange(100000, 100000 + n),
            "LOCAL": locales,
            "CODIGO CLIENTE": codigo_cliente,
            "RUN CLIENTE": [f"{10_000_000 + i}-{i % 10}" for i in range(n)],
            "NOMBRES": [f"NOMBRE_{i % 50}" for i in range(n)],
            "APELLIDOS": [f"APELLIDO_{i % 50}" for i in range(n)],
            "FECHA NACIMIENTO": fecha_nac,
            "GENERO": rng.integers(1, 3, size=n),
        }
    )
    return df


@pytest.fixture()
def synthetic_csv_path(tmp_path: Path, synthetic_raw_dataframe: pd.DataFrame) -> Path:
    """
    Escribe `synthetic_raw_dataframe` a un archivo CSV temporal
    delimitado por punto y coma (`;`), el formato real del archivo
    `ventas_completas.csv`, y retorna su ruta.

    Args:
        tmp_path: directorio temporal provisto por pytest (aislado por test).
        synthetic_raw_dataframe: fixture con los datos sintéticos.

    Returns:
        Path: ruta al CSV temporal generado.
    """
    from config import CSV_SEPARATOR

    csv_path = tmp_path / "ventas_completas_test.csv"
    synthetic_raw_dataframe.to_csv(csv_path, index=False, sep=CSV_SEPARATOR)
    return csv_path


@pytest.fixture()
def empty_csv_path(tmp_path: Path) -> Path:
    """Crea un archivo CSV vacío (0 bytes) para probar el manejo de errores."""
    path = tmp_path / "vacio.csv"
    path.touch()
    return path


@pytest.fixture()
def corrupt_csv_path(tmp_path: Path) -> Path:
    """Crea un archivo con contenido binario no parseable como CSV."""
    path = tmp_path / "corrupto.csv"
    path.write_bytes(bytes([0xFF, 0xFE, 0x00, 0x01, 0x02] * 20))
    return path


# =============================================================================
# Fixtures para la API REST (app/) — "Servicio ReST: Resumen estadístico"
# =============================================================================
@pytest.fixture(scope="session")
def synthetic_processed_dataframe() -> pd.DataFrame:
    """
    Construye un DataFrame sintético ya "post-pipeline" (con los mismos
    nombres de columna y dtypes que produce
    `src.data_loader.load_and_validate` + `src.data_cleaning.clean_dataset`
    + `src.feature_engineering.engineer_features` tras `.compute()`),
    para probar `app/services/estadisticas.py` y la API sin depender de
    Dask ni de un CSV real.

    Incluye deliberadamente valores nulos en GENERO y LOCAL (columnas de
    dtype nullable "Int64") para ejercitar el manejo de `pd.NA` en las
    máscaras de filtro (ver `app/services/estadisticas.py::calcular_estadisticas`).

    Returns:
        pd.DataFrame: 20 filas, 2 locales, 2 canales; GENERO cubre los 4
        códigos (0=No especificado x2, 1=Masculino x9, 2=Femenino x7,
        3=Otro x2), montos crecientes y determinísticos para que las
        estadísticas esperadas en los tests sean fáciles de calcular a
        mano.
    """
    n = 20
    df = pd.DataFrame(
        {
            "FECHA": pd.date_range("2026-01-01", periods=n, freq="D"),
            "CANAL": pd.Categorical(
                ["POS", "WEB"] * (n // 2), categories=["APP", "APR", "CCT", "POS", "WEB", "WPR"]
            ),
            "SKU": pd.array([1000 + i for i in range(n)], dtype="Int64"),
            "PRODUCTO": [f"PRODUCTO_{i}" for i in range(n)],
            "UNIDADES": pd.array([1] * n, dtype="Int64"),
            "PORCENTAJE_DESCUENTO": [0.1] * n,
            "MONTO_APLICADO": [float(100 * (i + 1)) for i in range(n)],
            "BOLETA": list(range(100000, 100000 + n)),
            "LOCAL": pd.array([1999] * 10 + [2000] * 10, dtype="Int64"),
            "CODIGO_CLIENTE": pd.array(
                [f"550e8400-e29b-41d4-a716-{i:012d}" for i in range(n)], dtype="string"
            ),
            "RUN_CLIENTE": [f"{10_000_000 + i}-{i % 10}" for i in range(n)],
            "NOMBRES": [f"NOMBRE_{i}" for i in range(n)],
            "APELLIDOS": [f"APELLIDO_{i}" for i in range(n)],
            "FECHA_NACIMIENTO": pd.date_range("1990-01-01", periods=n, freq="365D"),
            # 0,0, 3,3, 1x9, 2x7 (índices 0-1, 2-3, 4-12, 13-19)
            "GENERO": pd.array([0, 0, 3, 3] + [1] * 9 + [2] * 7, dtype="Int64"),
            "EDAD": [20.0 + (i % 5) for i in range(n)],
        }
    )
    # Nulos deliberados: una fila sin género (índice 5, dentro del bloque
    # GENERO=1) y otra sin local registrado.
    df.loc[5, "GENERO"] = pd.NA
    df.loc[1, "LOCAL"] = pd.NA
    return df


@pytest.fixture()
def api_client(monkeypatch: pytest.MonkeyPatch):
    """
    Cliente de pruebas de FastAPI (`TestClient`) para los tests que NO
    requieren datos reales: validación de los 8 filtros, formato exacto
    de error 400/500, documentación Swagger.

    Mediante `monkeypatch`, `ensure_schema_ready`/`ingest_if_needed`
    quedan como no-op y `db.get_connection` devuelve un objeto dummy en
    lugar de una conexión Postgres real. Esto es seguro porque la
    validación de filtros
    (`app/services/estadisticas.py::construir_clausula_where`) ocurre
    ANTES de tocar la base de datos: ningún test que solo ejercite rutas
    de error 400 llega a usar la conexión dummy. Los tests que necesitan
    resultados reales sobre datos (conteos, sumas) usan
    `api_client_integration` en su lugar.

    Yields:
        fastapi.testclient.TestClient
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import app.main as app_main

    monkeypatch.setattr(app_main.db, "ensure_schema_ready", lambda: None)
    monkeypatch.setattr(app_main, "ingest_if_needed", lambda csv_path: None)
    monkeypatch.setattr(app_main.db, "get_connection", lambda: object())

    with TestClient(app_main.app, raise_server_exceptions=False) as client:
        yield client


# =============================================================================
# Fixtures de integración con Postgres — se SALTAN automáticamente si
# CPYD_TEST_DATABASE_URL no está definida. La suite completa corre en
# cualquier máquina sin Postgres instalado; estos fixtures solo se activan
# cuando alguien sí tiene una base de pruebas configurada.
# =============================================================================
def pytest_configure(config: pytest.Config) -> None:  # noqa: D401 - hook de pytest
    config.addinivalue_line(
        "markers",
        "integration: requiere una base Postgres de pruebas real (CPYD_TEST_DATABASE_URL).",
    )


@pytest.fixture(scope="session")
def postgres_test_url() -> str:
    """URL de una base Postgres de pruebas. Salta el test si no está configurada."""
    from config import TEST_DATABASE_URL

    if not TEST_DATABASE_URL:
        pytest.skip(
            "CPYD_TEST_DATABASE_URL no está configurada; se omiten los tests "
            "de integración con Postgres."
        )
    return TEST_DATABASE_URL


@pytest.fixture()
def db_connection(postgres_test_url: str):
    """Conexión a la base de pruebas, con el esquema aplicado y `ventas`/`etl_status` reiniciados."""
    import src.db as db

    conexion = db.get_connection(postgres_test_url)
    db.ensure_schema(conexion)
    db.reiniciar_estado_carga(conexion)
    yield conexion
    conexion.close()


@pytest.fixture()
def api_client_integration(
    monkeypatch: pytest.MonkeyPatch,
    db_connection,
    postgres_test_url: str,
    synthetic_processed_dataframe: pd.DataFrame,
):
    """
    `TestClient` contra una base Postgres de pruebas REAL, poblada
    directamente con `synthetic_processed_dataframe` (evita repetir la
    ingesta de dos pasadas completa en cada test HTTP; el pipeline de
    ingesta en sí se prueba por separado en `tests/test_ingest_pipeline.py`).
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.db as db
    import app.main as app_main
    from app.services.dataset import _COLUMNAS_INGESTA

    columnas_presentes = [c for c in _COLUMNAS_INGESTA if c in synthetic_processed_dataframe.columns]
    pdf_db = synthetic_processed_dataframe[columnas_presentes].rename(columns=str.lower)
    db.ensure_month_partitions_for_chunk(db_connection, pdf_db, columna_fecha="fecha")
    db.copy_dataframe(db_connection, pdf_db)
    db.marcar_carga_completa(db_connection)

    monkeypatch.setattr(app_main.db, "ensure_schema_ready", lambda: None)
    monkeypatch.setattr(app_main, "ingest_if_needed", lambda csv_path: None)
    original_get_connection = db.get_connection
    monkeypatch.setattr(app_main.db, "get_connection", lambda: original_get_connection(postgres_test_url))

    with TestClient(app_main.app, raise_server_exceptions=False) as client:
        yield client
