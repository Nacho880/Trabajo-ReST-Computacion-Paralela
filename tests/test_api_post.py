"""
tests/test_api_post.py
========================

Pruebas de `POST /v1/estadisticas/ventas` (filtros vía body JSON).
Misma división que `test_api_get.py`: validación sin Postgres,
resultados reales como `@pytest.mark.integration`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

RUTA = "/v1/estadisticas/ventas"


# ---------------------------------------------------------------------------
# Sin Postgres: validación de estructura, filtros y formato de error.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("valor_consultas", [[], None])
def test_post_consultas_vacio_o_nulo_explicito_devuelve_400(api_client, valor_consultas) -> None:
    """
    `"consultas": []` o `"consultas": null` EXPLÍCITOS en el body son
    la condición "consultas vacío o nulo" que el enunciado lista como
    error 400 (a diferencia de un body `{}` o sin body, donde la clave
    `consultas` no aparece y sí se trata como "sin filtros").
    """
    respuesta = api_client.post(RUTA, json={"consultas": valor_consultas})
    assert respuesta.status_code == 400
    cuerpo = respuesta.json()
    assert cuerpo["errorCode"] == "VF"
    assert cuerpo["errorLabel"] == "Validación Fallida"
    assert cuerpo["method"] == "POST"


def test_post_consulta_no_permitida_devuelve_400(api_client) -> None:
    respuesta = api_client.post(
        RUTA, json={"consultas": [{"consulta": "TELEFONO", "valor": "123"}]}
    )
    assert respuesta.status_code == 400
    cuerpo = respuesta.json()
    assert cuerpo["errorCode"] == "VF"
    assert cuerpo["errorLabel"] == "Validación Fallida"
    assert cuerpo["method"] == "POST"


def test_post_valor_no_convertible_devuelve_400(api_client) -> None:
    respuesta = api_client.post(
        RUTA, json={"consultas": [{"consulta": "LOCAL", "valor": "qwerqwer"}]}
    )
    assert respuesta.status_code == 400
    assert respuesta.json()["errorCode"] == "VF"


def test_post_estructura_invalida_devuelve_400(api_client) -> None:
    """`consultas` no es una lista de objetos `{consulta, valor}`: 400, no 500."""
    respuesta = api_client.post(RUTA, json={"consultas": "no-es-una-lista"})
    assert respuesta.status_code == 400
    assert respuesta.json()["errorCode"] == "VF"


def test_post_elemento_sin_campo_valor_devuelve_400(api_client) -> None:
    respuesta = api_client.post(RUTA, json={"consultas": [{"consulta": "GENERO"}]})
    assert respuesta.status_code == 400
    assert respuesta.json()["errorCode"] == "VF"


# ---------------------------------------------------------------------------
# Con Postgres de pruebas: resultados reales sobre datos cargados.
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_post_sin_body_devuelve_estadisticas_globales(api_client_integration) -> None:
    """Sin body en absoluto equivale a "sin filtros", no es un error."""
    respuesta = api_client_integration.post(RUTA)
    assert respuesta.status_code == 200
    assert respuesta.json()["conteo"] == 20


@pytest.mark.integration
def test_post_body_vacio_sin_clave_consultas_equivale_a_sin_filtros(api_client_integration) -> None:
    """`{}` (la clave `consultas` no aparece) también es "sin filtros"."""
    respuesta = api_client_integration.post(RUTA, json={})
    assert respuesta.status_code == 200
    assert respuesta.json()["conteo"] == 20


@pytest.mark.integration
def test_post_ejemplo_del_enunciado_no_matchea_por_diseno(api_client_integration) -> None:
    """
    Reproduce EXACTAMENTE el body de ejemplo del enunciado (GENERO=Femenino,
    EDAD=31, CANAL=POS). No hay ninguna fila en el dataset sintético con
    EDAD=31, por lo que se espera conteo=0 (no un error): confirma que la
    combinación de los 3 filtros se aplica correctamente vía AND lógico.
    """
    respuesta = api_client_integration.post(
        RUTA,
        json={
            "consultas": [
                {"consulta": "GENERO", "valor": "Femenino"},
                {"consulta": "EDAD", "valor": "31"},
                {"consulta": "CANAL", "valor": "POS"},
            ]
        },
    )
    assert respuesta.status_code == 200
    assert respuesta.json()["conteo"] == 0


@pytest.mark.integration
def test_post_filtros_combinados_con_matches(api_client_integration) -> None:
    respuesta = api_client_integration.post(
        RUTA,
        json={
            "consultas": [
                {"consulta": "CANAL", "valor": "POS"},
                {"consulta": "LOCAL", "valor": "1999"},
            ]
        },
    )
    assert respuesta.status_code == 200
    assert respuesta.json()["conteo"] == 5
