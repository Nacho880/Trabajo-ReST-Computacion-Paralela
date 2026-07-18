"""
tests/test_ingest_pipeline.py
===============================

Prueba de integración end-to-end de `app/services/dataset.py::ingest_if_needed`:
CSV sintético pequeño -> ingesta de dos pasadas -> filas verificadas en
Postgres -> una segunda llamada no vuelve a insertar (idempotencia vía
`etl_status`).

Requiere una base Postgres de pruebas real (`CPYD_TEST_DATABASE_URL`);
se salta automáticamente si no está configurada.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_ingest_if_needed_carga_y_es_idempotente(
    db_connection, postgres_test_url, synthetic_csv_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.db as db
    from app.services import dataset as dataset_module

    # `ingest_if_needed` abre su propia conexión vía `db.get_connection()`;
    # se apunta a la misma base de pruebas que `db_connection` usa.
    original_get_connection = db.get_connection
    monkeypatch.setattr(
        dataset_module.db, "get_connection", lambda: original_get_connection(postgres_test_url)
    )

    dataset_module.ingest_if_needed(synthetic_csv_path)

    with db_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ventas")
        filas_tras_primera_carga = cur.fetchone()[0]
    assert filas_tras_primera_carga == 200  # tamaño de synthetic_raw_dataframe
    assert db.carga_completa(db_connection) is True

    # Segunda llamada: etl_status ya indica carga completa -> no debe
    # volver a insertar (idempotencia).
    dataset_module.ingest_if_needed(synthetic_csv_path)
    with db_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ventas")
        filas_tras_segunda_llamada = cur.fetchone()[0]
    assert filas_tras_segunda_llamada == filas_tras_primera_carga


@pytest.mark.integration
def test_particiones_mensuales_se_crean_segun_las_fechas_del_csv(
    db_connection, postgres_test_url, synthetic_csv_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    `synthetic_raw_dataframe` genera 200 filas horarias desde 2026-01-01,
    todas dentro de enero de 2026: debe existir exactamente la partición
    `ventas_y2026m01`, creada dinámicamente (no declarada en sql/ddl.sql).
    """
    import src.db as db
    from app.services import dataset as dataset_module

    original_get_connection = db.get_connection
    monkeypatch.setattr(
        dataset_module.db, "get_connection", lambda: original_get_connection(postgres_test_url)
    )
    dataset_module.ingest_if_needed(synthetic_csv_path)

    with db_connection.cursor() as cur:
        cur.execute("SELECT to_regclass('ventas_y2026m01') IS NOT NULL")
        assert cur.fetchone()[0] is True
