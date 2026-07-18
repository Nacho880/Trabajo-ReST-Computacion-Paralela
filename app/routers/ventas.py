"""
app/routers/ventas.py
=======================

Endpoints HTTP de `/v1/estadisticas/ventas` (GET y POST), tal como
exige el enunciado. Esta capa es puramente de traducción HTTP <->
dominio: parsea query params o body, delega el cálculo en
`app/services/estadisticas.py` y devuelve el resultado.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.exceptions import ErrorInternoError, ValidacionFallidaError
from app.models.schemas import ConsultasRequest, EstadisticaResponse
from app.services.estadisticas import FILTROS_PERMITIDOS, calcular_estadisticas
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/estadisticas", tags=["Estadísticas de ventas"])


def _obtener_conexion(request: Request):
    """
    Recupera la conexión Postgres abierta en el startup de la aplicación
    (`app.state.conexion`, ver `app/main.py`).

    Args:
        request: request HTTP en curso.

    Returns:
        conexión Postgres abierta.

    Raises:
        ErrorInternoError: si la conexión no está disponible (no debería
            ocurrir en operación normal, ya que el startup de FastAPI
            falla si `ingest_if_needed` falla; es una red de seguridad
            defensiva).

    Complejidad:
        O(1) tiempo y espacio.
    """
    conexion = getattr(request.app.state, "conexion", None)
    if conexion is None:
        raise ErrorInternoError(
            "La conexión a la base de datos aún no está disponible en el servidor."
        )
    return conexion


@router.get(
    "/ventas",
    response_model=EstadisticaResponse,
    summary="Estadísticas de ventas con filtros predeterminados (query params)",
    description=(
        "Devuelve suma, conteo, promedio, mínimo, máximo, mediana y "
        "desviación estándar de las ventas, opcionalmente filtradas por "
        "cualquier combinación de: " + ", ".join(FILTROS_PERMITIDOS) + "."
    ),
)
def obtener_estadisticas_get(
    request: Request,
    GENERO: str | None = Query(
        default=None, description="No especificado | Masculino | Femenino | Otro"
    ),
    EDAD: str | None = Query(default=None, description="Entero, ej. 31"),
    CANAL: str | None = Query(default=None, description="POS | WEB | APP | CCT | APR | WPR"),
    CODIGO_PRODUCTO: str | None = Query(default=None, description="SKU, entero"),
    ID_PERSONA: str | None = Query(default=None, description="UUID del cliente"),
    LOCAL: str | None = Query(default=None, description="Número de local, entero"),
    FECHA_DESDE: str | None = Query(default=None, description="ISO-8601, ej. 2026-01-01T00:00:00"),
    FECHA_HASTA: str | None = Query(default=None, description="ISO-8601, ej. 2026-06-30T23:59:59"),
) -> EstadisticaResponse:
    """
    `GET /v1/estadisticas/ventas`: filtros opcionales vía query params.

    Complejidad:
        Ver `calcular_estadisticas`.
    """
    valores_recibidos = {
        "GENERO": GENERO,
        "EDAD": EDAD,
        "CANAL": CANAL,
        "CODIGO_PRODUCTO": CODIGO_PRODUCTO,
        "ID_PERSONA": ID_PERSONA,
        "LOCAL": LOCAL,
        "FECHA_DESDE": FECHA_DESDE,
        "FECHA_HASTA": FECHA_HASTA,
    }
    filtros = [(nombre, valor) for nombre, valor in valores_recibidos.items() if valor is not None]

    df = _obtener_conexion(request)
    resultado = calcular_estadisticas(df, filtros)
    return EstadisticaResponse(**resultado)


@router.post(
    "/ventas",
    response_model=EstadisticaResponse,
    summary="Estadísticas de ventas con filtros personalizados (body JSON)",
    description=(
        "Devuelve suma, conteo, promedio, mínimo, máximo, mediana y "
        "desviación estándar de las ventas, opcionalmente filtradas por "
        "cualquier combinación de: " + ", ".join(FILTROS_PERMITIDOS) + ". "
        "Si el body es `{}` o se omite por completo (la clave `consultas` "
        "no aparece), se calculan las estadísticas sobre TODO el dataset "
        "(equivalente a un GET sin query params). En cambio, enviar "
        "`\"consultas\": null` o `\"consultas\": []` de forma EXPLÍCITA "
        "es un error 400, tal como exige el enunciado (\"consultas vacío "
        "o nulo\")."
    ),
)
def obtener_estadisticas_post(
    request: Request,
    payload: ConsultasRequest = ConsultasRequest(),
) -> EstadisticaResponse:
    """
    `POST /v1/estadisticas/ventas`: filtros opcionales vía body JSON,
    `{"consultas": [{"consulta": "GENERO", "valor": "Femenino"}, ...]}`.

    El enunciado exige 400 para "consultas vacío o nulo", pero también
    dice que "las consultas pueden realizarse sin usar estos filtros".
    Se resuelve la aparente contradicción distinguiendo CÓMO se pide
    "sin filtros":
        - Body ausente, o `{}` (la clave `consultas` no aparece en el
          JSON): "sin usar estos filtros" legítimo -> 200, sin
          filtrar.
        - `"consultas": null` o `"consultas": []` EXPLÍCITOS (la clave
          SÍ aparece, con esos valores): esto es literalmente
          "consultas ... vacío o nulo" -> 400 "Validación Fallida".

    Se distingue "ausente" de "presente con ese valor" usando
    `payload.model_fields_set`, que Pydantic solo puebla con
    `"consultas"` si el cliente envió esa clave en el JSON (aunque su
    valor sea `null` o `[]`).

    Raises:
        ValidacionFallidaError: si `"consultas": null` o
            `"consultas": []` fue enviado explícitamente en el body.

    Complejidad:
        Ver `calcular_estadisticas`.
    """
    if "consultas" in payload.model_fields_set and not payload.consultas:
        raise ValidacionFallidaError(
            "El campo 'consultas' no puede ser nulo ni una lista vacía. "
            "Omita el campo por completo (o envíe un body `{}`) para "
            "consultar sin filtros, o envíe una lista con uno o más "
            "filtros."
        )
    filtros = payload.filtros_como_lista()
    df = _obtener_conexion(request)
    resultado = calcular_estadisticas(df, filtros)
    return EstadisticaResponse(**resultado)
