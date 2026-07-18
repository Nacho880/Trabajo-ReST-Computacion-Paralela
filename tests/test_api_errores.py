"""
tests/test_api_errores.py
===========================

Prueba el formato exacto de error 500 exigido por el enunciado ante un
fallo inesperado (no de validación) durante el cálculo de estadísticas,
simulado forzando una excepción dentro del router.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi")

RUTA = "/v1/estadisticas/ventas"

_PATRON_TIMESTAMP_9_DIGITOS = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$"
)


def test_timestamp_error_tiene_formato_exacto_del_enunciado(api_client) -> None:
    """
    El `timestamp` debe tener EXACTAMENTE el mismo formato del ejemplo
    del enunciado: `"2026-06-30T20:44:49.201437123Z"` — 9 dígitos
    decimales de fracción de segundo (estilo nanosegundos) + sufijo Z.
    """
    respuesta = api_client.post(RUTA, json={"consultas": None})
    assert respuesta.status_code == 400
    timestamp = respuesta.json()["timestamp"]
    assert _PATRON_TIMESTAMP_9_DIGITOS.match(timestamp), timestamp



def test_error_inesperado_devuelve_500_formato_exacto(api_client, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.routers.ventas as ventas_router

    def _falla(df, filtros):
        raise RuntimeError("Error al calcular la desviación estándar")

    monkeypatch.setattr(ventas_router, "calcular_estadisticas", _falla)

    respuesta = api_client.get(RUTA)

    assert respuesta.status_code == 500
    cuerpo = respuesta.json()
    assert cuerpo["status"] == 500
    assert cuerpo["title"] == "Internal Server Error"
    assert cuerpo["errorCode"] == "IE"
    assert cuerpo["errorLabel"] == "Error Interno"
    assert cuerpo["instance"] == RUTA
    assert cuerpo["method"] == "GET"
    assert cuerpo["type"] == "https://developer.mozilla.org/es/docs/Web/HTTP/Reference/Status/500"
    assert set(cuerpo.keys()) == {
        "detail", "instance", "status", "title", "type", "timestamp",
        "errorCode", "errorLabel", "method",
    }
