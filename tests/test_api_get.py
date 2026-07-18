"""
tests/test_api_get.py
=======================

Pruebas de `GET /v1/estadisticas/ventas` (filtros vía query params).

Se dividen en dos grupos, según si necesitan datos reales:
    - Validación de filtros / formato de error / documentación: no
      requieren Postgres (`api_client`).
    - Conteos/estadísticas reales sobre datos: requieren una base
      Postgres de pruebas real (`api_client_integration`,
      `@pytest.mark.integration`, se saltan automáticamente sin
      `CPYD_TEST_DATABASE_URL`).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

RUTA = "/v1/estadisticas/ventas"


# ---------------------------------------------------------------------------
# Sin Postgres: validación de filtros, formato de error, documentación.
# ---------------------------------------------------------------------------
def test_get_filtro_edad_invalido_devuelve_400_formato_exacto(api_client) -> None:
    respuesta = api_client.get(RUTA, params={"EDAD": "qwerqwer"})
    assert respuesta.status_code == 400
    cuerpo = respuesta.json()
    assert cuerpo["status"] == 400
    assert cuerpo["title"] == "Bad Request"
    assert cuerpo["errorCode"] == "VF"
    assert cuerpo["errorLabel"] == "Validación Fallida"
    assert cuerpo["instance"] == RUTA
    assert cuerpo["method"] == "GET"
    assert cuerpo["type"] == "https://developer.mozilla.org/es/docs/Web/HTTP/Reference/Status/400"
    assert "timestamp" in cuerpo and "detail" in cuerpo
    assert set(cuerpo.keys()) == {
        "detail", "instance", "status", "title", "type", "timestamp",
        "errorCode", "errorLabel", "method",
    }


def test_get_filtro_canal_invalido_devuelve_400(api_client) -> None:
    respuesta = api_client.get(RUTA, params={"CANAL": "TELEFONO"})
    assert respuesta.status_code == 400
    assert respuesta.json()["errorCode"] == "VF"


def test_get_documentacion_swagger_disponible(api_client) -> None:
    respuesta = api_client.get("/docs")
    assert respuesta.status_code == 200
    respuesta_schema = api_client.get("/openapi.json")
    assert respuesta_schema.status_code == 200
    assert RUTA in respuesta_schema.json()["paths"]


# ---------------------------------------------------------------------------
# Con Postgres de pruebas: estadísticas reales sobre datos cargados.
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_get_sin_filtros_devuelve_estadisticas_globales(api_client_integration) -> None:
    respuesta = api_client_integration.get(RUTA)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo.keys()) == {
        "suma", "conteo", "promedio", "minimo", "maximo", "mediana", "desviacion_estandar",
    }
    assert cuerpo["conteo"] == 20


@pytest.mark.integration
def test_get_con_filtro_genero(api_client_integration) -> None:
    respuesta = api_client_integration.get(RUTA, params={"GENERO": "Femenino"})
    assert respuesta.status_code == 200
    assert respuesta.json()["conteo"] == 7


@pytest.mark.integration
def test_get_con_multiples_filtros_combinados(api_client_integration) -> None:
    respuesta = api_client_integration.get(RUTA, params={"CANAL": "POS", "LOCAL": "1999"})
    assert respuesta.status_code == 200
    assert respuesta.json()["conteo"] == 5
