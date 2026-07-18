"""
tests/test_estadisticas_service.py
====================================

Pruebas de `app/services/estadisticas.py`.

Se dividen en dos grupos:
    - `construir_clausula_where`: función PURA (sin base de datos) que
      valida cada uno de los 8 filtros y arma el SQL parametrizado. Se
      prueba exhaustivamente aquí, sin ninguna dependencia de Postgres
      — mismos casos y mismos mensajes de error que la versión anterior
      basada en máscaras de Pandas.
    - `calcular_estadisticas` (consulta real a Postgres): se prueba
      como test de integración (`@pytest.mark.integration`), poblando
      una base de pruebas real con `synthetic_processed_dataframe` y
      verificando los mismos conteos que antes.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi")

from app.exceptions import ValidacionFallidaError  # noqa: E402
from app.services.estadisticas import (  # noqa: E402
    FILTROS_PERMITIDOS,
    calcular_estadisticas,
    construir_clausula_where,
)


# ---------------------------------------------------------------------------
# construir_clausula_where: validación pura de los 8 filtros.
# ---------------------------------------------------------------------------
def test_sin_filtros_devuelve_clausula_vacia() -> None:
    where_sql, params = construir_clausula_where([])
    assert where_sql == ""
    assert params == []


@pytest.mark.parametrize(
    "valor,codigo_esperado",
    [("No especificado", 0), ("Masculino", 1), ("Femenino", 2), ("Otro", 3)],
)
def test_filtro_genero_mapea_texto_a_codigo(valor: str, codigo_esperado: int) -> None:
    where_sql, params = construir_clausula_where([("GENERO", valor)])
    assert where_sql == "genero = %s"
    assert params == [codigo_esperado]


def test_filtro_genero_valor_invalido_lanza_400() -> None:
    with pytest.raises(ValidacionFallidaError):
        construir_clausula_where([("GENERO", "no-existe")])


def test_filtro_edad_entero_valido() -> None:
    where_sql, params = construir_clausula_where([("EDAD", "31")])
    assert where_sql == "edad = %s"
    assert params == [31]


def test_filtro_edad_valor_no_entero_lanza_400() -> None:
    with pytest.raises(ValidacionFallidaError, match="entero válido"):
        construir_clausula_where([("EDAD", "qwerqwer")])


def test_filtro_canal_valido() -> None:
    where_sql, params = construir_clausula_where([("CANAL", "POS")])
    assert where_sql == "canal = %s"
    assert params == ["POS"]


def test_filtro_canal_case_insensitive() -> None:
    where_sql, params = construir_clausula_where([("CANAL", "pos")])
    assert params == ["POS"]


def test_filtro_canal_invalido_lanza_400() -> None:
    with pytest.raises(ValidacionFallidaError, match="canal válido"):
        construir_clausula_where([("CANAL", "TELEFONO")])


def test_filtro_codigo_producto_valido() -> None:
    where_sql, params = construir_clausula_where([("CODIGO_PRODUCTO", "1005")])
    assert where_sql == "sku = %s"
    assert params == [1005]


def test_filtro_codigo_producto_invalido_lanza_400() -> None:
    with pytest.raises(ValidacionFallidaError):
        construir_clausula_where([("CODIGO_PRODUCTO", "abc")])


def test_filtro_id_persona_uuid_valido() -> None:
    where_sql, params = construir_clausula_where(
        [("ID_PERSONA", "550E8400-E29B-41D4-A716-000000000003")]
    )
    assert where_sql == "lower(codigo_cliente) = %s"
    assert params == ["550e8400-e29b-41d4-a716-000000000003"]


def test_filtro_id_persona_uuid_invalido_lanza_400() -> None:
    with pytest.raises(ValidacionFallidaError, match="UUID válido"):
        construir_clausula_where([("ID_PERSONA", "no-es-un-uuid")])


def test_filtro_local_valido() -> None:
    where_sql, params = construir_clausula_where([("LOCAL", "1999")])
    assert where_sql == "local = %s"
    assert params == [1999]


def test_filtro_local_invalido_lanza_400() -> None:
    with pytest.raises(ValidacionFallidaError, match="ID de tienda"):
        construir_clausula_where([("LOCAL", "qwerqwer")])


def test_filtro_rango_fechas_genera_dos_condiciones() -> None:
    where_sql, params = construir_clausula_where(
        [("FECHA_DESDE", "2026-01-05"), ("FECHA_HASTA", "2026-01-10")]
    )
    assert where_sql == "fecha >= %s AND fecha <= %s"
    assert params == [pd.Timestamp("2026-01-05").to_pydatetime(), pd.Timestamp("2026-01-10").to_pydatetime()]


def test_filtro_fecha_invalida_lanza_400() -> None:
    with pytest.raises(ValidacionFallidaError, match="ISO-8601"):
        construir_clausula_where([("FECHA_DESDE", "05/01/2026")])


def test_combinacion_de_filtros_se_une_con_and() -> None:
    where_sql, params = construir_clausula_where([("CANAL", "POS"), ("LOCAL", "1999")])
    assert where_sql == "canal = %s AND local = %s"
    assert params == ["POS", 1999]


def test_filtro_no_permitido_lanza_400() -> None:
    with pytest.raises(ValidacionFallidaError, match="no es uno de los valores permitidos"):
        construir_clausula_where([("NO_EXISTE", "1")])


def test_filtros_permitidos_son_exactamente_los_ocho_del_enunciado() -> None:
    assert set(FILTROS_PERMITIDOS) == {
        "GENERO", "EDAD", "CANAL", "CODIGO_PRODUCTO",
        "ID_PERSONA", "LOCAL", "FECHA_DESDE", "FECHA_HASTA",
    }


# ---------------------------------------------------------------------------
# calcular_estadisticas contra Postgres real (integración).
# ---------------------------------------------------------------------------
@pytest.fixture()
def _conexion_con_datos(db_connection, synthetic_processed_dataframe: pd.DataFrame):
    import src.db as db
    from app.services.dataset import _COLUMNAS_INGESTA

    columnas = [c for c in _COLUMNAS_INGESTA if c in synthetic_processed_dataframe.columns]
    pdf_db = synthetic_processed_dataframe[columnas].rename(columns=str.lower)
    db.ensure_month_partitions_for_chunk(db_connection, pdf_db, columna_fecha="fecha")
    db.copy_dataframe(db_connection, pdf_db)
    return db_connection


@pytest.mark.integration
def test_sin_filtros_calcula_sobre_todo_el_dataset(_conexion_con_datos) -> None:
    resultado = calcular_estadisticas(_conexion_con_datos, [])
    assert resultado["conteo"] == 20
    assert resultado["promedio"] == pytest.approx(resultado["suma"] / 20)
    assert resultado["minimo"] == pytest.approx(100.0)
    assert resultado["maximo"] == pytest.approx(2000.0)


@pytest.mark.integration
def test_conjunto_vacio_devuelve_ceros_sin_error(_conexion_con_datos) -> None:
    resultado = calcular_estadisticas(_conexion_con_datos, [("LOCAL", "555555")])
    assert resultado == {
        "suma": 0.0, "conteo": 0, "promedio": 0.0, "minimo": 0.0,
        "maximo": 0.0, "mediana": 0.0, "desviacion_estandar": 0.0,
    }


@pytest.mark.integration
def test_filtro_genero_no_ignora_filas_con_nulo(_conexion_con_datos) -> None:
    """La fila con GENERO nulo (fixture) no debe matchear ningún valor de GENERO."""
    total = sum(
        calcular_estadisticas(_conexion_con_datos, [("GENERO", valor)])["conteo"]
        for valor in ("No especificado", "Masculino", "Femenino", "Otro")
    )
    assert total == 19  # 20 - 1 fila con GENERO nulo


@pytest.mark.integration
def test_filtro_local_valido_excluye_nulos(_conexion_con_datos) -> None:
    """La fixture deja una fila con LOCAL=1999 en LOCAL=NULL: debe excluirse, no romper."""
    resultado = calcular_estadisticas(_conexion_con_datos, [("LOCAL", "1999")])
    assert resultado["conteo"] == 9  # 10 filas con LOCAL=1999, una es nula


@pytest.mark.integration
def test_combinacion_de_filtros_aplica_and(_conexion_con_datos) -> None:
    resultado = calcular_estadisticas(_conexion_con_datos, [("CANAL", "POS"), ("LOCAL", "1999")])
    assert resultado["conteo"] == 5
