"""
tests/test_db.py
==================

Pruebas de `src/db.py`.

- `nombre_particion_para_fecha`: función PURA, sin conexión — se prueba
  exhaustivamente aquí (incluye el caso de "particionado completamente
  dinámico": ninguna fecha de prueba está hardcodeada en `src/db.py`,
  el nombre/rango se deriva por completo del valor recibido).
- El resto (`ensure_schema`, `ensure_month_partition`, `copy_dataframe`,
  `etl_status`) requiere una base Postgres real: tests de integración,
  se saltan automáticamente sin `CPYD_TEST_DATABASE_URL`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.db import nombre_particion_para_fecha


# ---------------------------------------------------------------------------
# nombre_particion_para_fecha: pura, sin DB.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fecha,nombre_esperado,inicio_esperado,fin_esperado",
    [
        ("2026-05-08", "ventas_y2026m05", "2026-05-01", "2026-06-01"),
        ("2026-12-31T23:59:59", "ventas_y2026m12", "2026-12-01", "2027-01-01"),
        ("2030-01-01", "ventas_y2030m01", "2030-01-01", "2030-02-01"),
        ("1998-02-14", "ventas_y1998m02", "1998-02-01", "1998-03-01"),
    ],
)
def test_nombre_particion_para_fecha_es_completamente_dinamico(
    fecha: str, nombre_esperado: str, inicio_esperado: str, fin_esperado: str
) -> None:
    """
    Ninguno de estos años/meses está hardcodeado en `src/db.py`: el
    nombre y el rango [inicio, fin) se calculan a partir de la fecha
    recibida, cualquiera que esta sea.
    """
    nombre, inicio, fin = nombre_particion_para_fecha(fecha)
    assert nombre == nombre_esperado
    assert inicio == pd.Timestamp(inicio_esperado)
    assert fin == pd.Timestamp(fin_esperado)


def test_nombre_particion_rango_es_de_un_mes_exacto() -> None:
    _, inicio, fin = nombre_particion_para_fecha("2026-07-17")
    assert fin == inicio + pd.DateOffset(months=1)


# ---------------------------------------------------------------------------
# get_connection: mensaje de error amigable sin URL configurada (pura,
# no requiere Postgres real).
# ---------------------------------------------------------------------------
def test_get_connection_sin_url_lanza_error_explicativo(monkeypatch: pytest.MonkeyPatch) -> None:
    import config
    from src.db import get_connection

    monkeypatch.setattr(config, "DATABASE_URL", "")

    with pytest.raises(RuntimeError) as excinfo:
        get_connection()

    mensaje = str(excinfo.value)
    assert "No existe una conexión PostgreSQL configurada" in mensaje
    assert "CPYD_DATABASE_URL" in mensaje
    assert "postgresql://usuario:password@host:5432/base_datos" in mensaje


# ---------------------------------------------------------------------------
# _render_ddl: CPYD_CREATE_INDEXES controla si se incluyen los CREATE
# INDEX del esquema (pura, no requiere Postgres real).
# ---------------------------------------------------------------------------
def test_render_ddl_incluye_indices_por_defecto() -> None:
    import config
    from src.db import _render_ddl

    assert config.CREATE_INDEXES is True
    ddl = _render_ddl()
    assert "CREATE INDEX" in ddl
    assert "CREATE TABLE IF NOT EXISTS ventas" in ddl


def test_render_ddl_omite_indices_si_create_indexes_es_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import config
    from src.db import _render_ddl

    monkeypatch.setattr(config, "CREATE_INDEXES", False)
    ddl = _render_ddl()
    sentencias_create_index = [
        linea for linea in ddl.splitlines()
        if linea.strip().upper().startswith("CREATE INDEX")
    ]
    assert sentencias_create_index == []
    # Las tablas base deben seguir presentes: solo se omiten los índices.
    assert "CREATE TABLE IF NOT EXISTS ventas" in ddl
    assert "CREATE TABLE IF NOT EXISTS etl_status" in ddl


# ---------------------------------------------------------------------------
# Integración: requiere una base Postgres de pruebas real.
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_ensure_schema_es_idempotente(db_connection) -> None:
    import src.db as db

    db.ensure_schema(db_connection)  # segunda ejecución, no debe fallar
    assert db.carga_completa(db_connection) is False


@pytest.mark.integration
def test_ensure_month_partition_crea_particion_dinamica(db_connection) -> None:
    import src.db as db

    nombre = db.ensure_month_partition(db_connection, "2031-03-15")
    assert nombre == "ventas_y2031m03"
    with db_connection.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (nombre,))
        assert cur.fetchone()[0] is True


# ---------------------------------------------------------------------------
# copy_dataframe: manejo de DiskFull (pura, con conexión/cursor simulados
# -- no requiere Postgres real ni espacio en disco realmente agotado).
# ---------------------------------------------------------------------------
def test_copy_dataframe_disco_lleno_lanza_runtimeerror_explicativo() -> None:
    import psycopg2
    from src.db import copy_dataframe

    df = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2026-04-10"]),
            "canal": ["POS"],
            "monto_aplicado": [1000.0],
        }
    )

    class _CursorDiskFull:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def copy_expert(self, *args, **kwargs):
            raise psycopg2.errors.DiskFull("could not extend file")

    class _ConexionDiskFull:
        def cursor(self):
            return _CursorDiskFull()

        def commit(self):
            raise AssertionError("no debería llegar a hacer commit")

        def rollback(self):
            pass

    with pytest.raises(RuntimeError) as excinfo:
        copy_dataframe(_ConexionDiskFull(), df)

    assert "espacio suficiente" in str(excinfo.value)


@pytest.mark.integration
def test_copy_dataframe_inserta_filas(db_connection) -> None:
    import src.db as db

    df = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2026-04-10", "2026-04-11"]),
            "canal": ["POS", "WEB"],
            "monto_aplicado": [1000.0, 2000.0],
        }
    )
    db.ensure_month_partitions_for_chunk(db_connection, df, columna_fecha="fecha")
    filas = db.copy_dataframe(db_connection, df)
    assert filas == 2
    with db_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ventas")
        assert cur.fetchone()[0] == 2


@pytest.mark.integration
def test_etl_status_marca_carga_completa(db_connection) -> None:
    import src.db as db

    assert db.carga_completa(db_connection) is False
    db.marcar_carga_completa(db_connection)
    assert db.carga_completa(db_connection) is True
