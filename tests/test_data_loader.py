"""
tests/test_data_loader.py
==========================

Pruebas unitarias y funcionales para `src/data_loader.py`.

Las pruebas que requieren `dask` real (carga completa) se marcan con
`pytest.importorskip("dask")`, de modo que la suite completa siga
siendo ejecutable en entornos donde dask aún no esté instalado,
sin dar falsos positivos ni falsos negativos: simplemente se omiten
con un mensaje explícito.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_loader import load_and_validate
from src.utils.validators import (
    CorruptDataError,
    FileNotFoundOrEmptyError,
    SchemaValidationError,
)


def test_load_and_validate_raises_on_missing_file(tmp_path: Path) -> None:
    """Debe lanzar FileNotFoundOrEmptyError si el archivo no existe."""
    missing_path = tmp_path / "no_existe.csv"
    with pytest.raises(FileNotFoundOrEmptyError):
        load_and_validate(missing_path)


def test_load_and_validate_raises_on_empty_file(empty_csv_path: Path) -> None:
    """Debe lanzar FileNotFoundOrEmptyError si el archivo tiene 0 bytes."""
    with pytest.raises(FileNotFoundOrEmptyError):
        load_and_validate(empty_csv_path)


def test_load_and_validate_raises_on_corrupt_file(corrupt_csv_path: Path) -> None:
    """Debe lanzar CorruptDataError si el archivo no es un CSV parseable."""
    with pytest.raises(CorruptDataError):
        load_and_validate(corrupt_csv_path)


def test_load_and_validate_raises_on_missing_columns(tmp_path: Path) -> None:
    """Debe lanzar SchemaValidationError si faltan columnas obligatorias."""
    incomplete_path = tmp_path / "incompleto.csv"
    incomplete_path.write_text("FECHA;CANAL\n2026-01-01;POS\n")
    with pytest.raises(SchemaValidationError):
        load_and_validate(incomplete_path)


def test_validation_sample_and_schema_pass_for_well_formed_csv(
    synthetic_csv_path: Path,
) -> None:
    """
    Verifica, SIN depender de dask, que un CSV bien formado:
        1. Es leído correctamente por `_read_validation_sample`.
        2. Pasa `validate_schema` sin lanzar excepciones.

    Esto aísla la lógica de validación (que no requiere dask) de la
    lógica de lectura perezosa (que sí lo requiere), permitiendo
    detectar bugs de validación aunque dask no esté instalado.
    """
    from config import RAW_EXPECTED_COLUMNS
    from src.data_loader import _read_validation_sample
    from src.utils.validators import validate_schema

    sample = _read_validation_sample(synthetic_csv_path)
    assert len(sample) == 200

    # No debe lanzar excepción: el esquema sintético es válido.
    validate_schema(list(sample.columns), RAW_EXPECTED_COLUMNS)


def test_load_and_validate_with_dask(synthetic_csv_path: Path) -> None:
    """
    Prueba de integración completa: carga real con Dask.

    Se omite automáticamente si `dask` no está instalado en el entorno
    de ejecución (por ejemplo, en el sandbox de desarrollo sin acceso a
    red). Debe ejecutarse en el entorno final del estudiante, donde
    `requirements.txt` habrá instalado dask.
    """
    pytest.importorskip("dask")
    from config import STANDARD_COLUMNS

    ddf = load_and_validate(synthetic_csv_path)

    assert ddf.npartitions >= 1
    for column in STANDARD_COLUMNS:
        assert column in ddf.columns

    computed = ddf.compute()
    assert len(computed) == 200
    assert computed["MONTO_APLICADO"].max() > 0
