"""
app/exceptions.py
==================

Excepciones de dominio de la API y su traducción a la respuesta JSON de
error EXACTA exigida por el enunciado ("Servicio ReST: Resumen
estadístico"):

    {
      "detail": "...",
      "instance": "/v1/estadisticas/ventas",
      "status": 400,
      "title": "Bad Request",
      "type": "https://developer.mozilla.org/es/docs/Web/HTTP/Reference/Status/400",
      "timestamp": "2026-06-30T20:44:49.201437123Z",
      "errorCode": "VF",
      "errorLabel": "Validación Fallida",
      "method": "POST"
    }

Diseño: se centraliza aquí TODA la lógica de construcción del cuerpo de
error (nombres de campo exactos, sin agregar ni renombrar ninguno, tal
como exige el enunciado), para que ningún endpoint tenga que armar ese
diccionario a mano y arriesgarse a desviarse del formato.

No se reutiliza `src/utils/validators.py` para esto: esas excepciones
(`DataValidationError` y subclases) son errores de **carga del archivo**
en tiempo de arranque, con un dominio de mensajes distinto al de errores
de **request HTTP** (parámetros/filtros de una consulta puntual), que se
modelan aquí.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MDN_STATUS_URL = "https://developer.mozilla.org/es/docs/Web/HTTP/Reference/Status/{status}"


class APIError(Exception):
    """
    Excepción base para todo error de dominio de la API que deba
    traducirse a la respuesta JSON exacta del enunciado.

    Attributes:
        detail: descripción legible del error (ej. "El valor 'X' no es
            válido para Y").
        status: código de estado HTTP (400 o 500 en este enunciado).
        title: título estándar del estado HTTP ("Bad Request",
            "Internal Server Error").
        error_code: código corto exigido por el enunciado ("VF", "IE").
        error_label: etiqueta legible exigida por el enunciado
            ("Validación Fallida", "Error Interno").
    """

    status: int = 500
    title: str = "Internal Server Error"
    error_code: str = "IE"
    error_label: str = "Error Interno"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ValidacionFallidaError(APIError):
    """
    Error 400: la consulta (query params o body POST) no cumple el
    contrato del enunciado.

    Se lanza en exactamente los tres casos que exige el enunciado:
        - `consultas` presente pero mal formado (no es una lista
          utilizable, o algún elemento no tiene la forma
          `{"consulta": ..., "valor": ...}`).
        - `consulta` no es uno de los 8 filtros permitidos.
        - `valor` no es convertible al tipo esperado por ese filtro.

    Nota sobre "consultas vacío o nulo": el enunciado lista esto como
    condición de error 400, pero también establece explícitamente que
    "las consultas pueden realizarse sin usar estos filtros". Ambas
    afirmaciones son estrictamente contradictorias si "vacío" incluye
    el caso de cero filtros. La interpretación adoptada (documentada en
    detalle en README.md, sección "Decisiones y supuestos") es que una
    lista vacía o ausente de `consultas` es el mecanismo LEGÍTIMO para
    pedir estadísticas sin filtrar (equivalente a un GET sin query
    params), y que el error 400 aplica solo cuando `consultas` está
    presente pero es estructuralmente inválida (ej. no es una lista, o
    trae un elemento sin los campos requeridos).
    """

    status = 400
    title = "Bad Request"
    error_code = "VF"
    error_label = "Validación Fallida"


class ErrorInternoError(APIError):
    """
    Error 500: fallo inesperado durante el cálculo de estadísticas
    (ej. error al calcular la desviación estándar sobre datos
    corruptos en memoria), no atribuible a una consulta mal formada.
    """

    status = 500
    title = "Internal Server Error"
    error_code = "IE"
    error_label = "Error Interno"


def _timestamp_iso_9_digitos() -> str:
    """
    Genera el timestamp UTC actual en el mismo formato EXACTO del
    ejemplo del enunciado: ISO-8601 con 9 dígitos decimales de
    fracción de segundo (estilo nanosegundos) y sufijo `Z`, ej.
    `"2026-06-30T20:44:49.201437123Z"`.

    Python (`datetime`) solo ofrece precisión de microsegundos (6
    dígitos) de forma nativa; no hay reloj de sistema portable con
    resolución real de nanosegundos accesible desde `datetime`. Se
    obtienen los 6 dígitos reales de precisión (`%f`) y se rellenan
    con 3 ceros a la derecha para igualar el formato visual de 9
    dígitos del ejemplo, sin inventar precisión que no se tiene.

    Returns:
        str: timestamp con exactamente 9 dígitos de fracción de
        segundo, ej. `"2026-07-16T21:38:39.360090000Z"`.

    Complejidad:
        O(1) tiempo y espacio.
    """
    ahora = datetime.now(timezone.utc)
    microsegundos = ahora.strftime("%f")  # 6 dígitos
    return ahora.strftime("%Y-%m-%dT%H:%M:%S") + f".{microsegundos}000Z"


def _build_error_body(
    *, detail: str, status: int, title: str, error_code: str, error_label: str,
    instance: str, method: str,
) -> dict:
    """
    Construye el diccionario de error con exactamente los campos y
    nombres exigidos por el enunciado, ni uno más ni uno menos.

    Args:
        detail: descripción legible del error.
        status: código de estado HTTP.
        title: título estándar del estado HTTP.
        error_code: código corto ("VF" o "IE").
        error_label: etiqueta legible del error.
        instance: ruta del endpoint que originó el error.
        method: método HTTP de la request que originó el error.

    Returns:
        dict: cuerpo de error exacto, listo para `JSONResponse`.

    Complejidad:
        O(1) tiempo y espacio.
    """
    return {
        "detail": detail,
        "instance": instance,
        "status": status,
        "title": title,
        "type": _MDN_STATUS_URL.format(status=status),
        "timestamp": _timestamp_iso_9_digitos(),
        "errorCode": error_code,
        "errorLabel": error_label,
        "method": method,
    }


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """
    Maneja cualquier `APIError` (y subclases) lanzada dentro de un
    endpoint, traduciéndola a la respuesta JSON exacta del enunciado.

    Args:
        request: request HTTP que originó el error (se usa para
            `instance` y `method`).
        exc: excepción de dominio capturada.

    Returns:
        JSONResponse: respuesta con el código de estado y cuerpo
        exactos exigidos por el enunciado.

    Complejidad:
        O(1) tiempo y espacio.
    """
    if exc.status >= 500:
        logger.error("Error interno en %s %s: %s", request.method, request.url.path, exc.detail)
    else:
        logger.warning(
            "Validación fallida en %s %s: %s", request.method, request.url.path, exc.detail
        )
    body = _build_error_body(
        detail=exc.detail,
        status=exc.status,
        title=exc.title,
        error_code=exc.error_code,
        error_label=exc.error_label,
        instance=request.url.path,
        method=request.method,
    )
    return JSONResponse(status_code=exc.status, content=body)


async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Reformatea los errores de validación AUTOMÁTICOS de FastAPI/Pydantic
    (ej. `POST` con un body cuyo JSON es sintácticamente inválido, o
    donde `valor`/`consulta` no son strings) al mismo formato JSON exacto
    de error 400 del enunciado, en lugar de dejar pasar el formato
    `{"detail": [...]}` por defecto de FastAPI (que no cumple el
    contrato exigido).

    Args:
        request: request HTTP que originó el error.
        exc: excepción de validación capturada por FastAPI.

    Returns:
        JSONResponse: error 400 en el formato exacto del enunciado.

    Complejidad:
        O(k) tiempo, k = número de errores de validación reportados por
        Pydantic (típicamente 1); O(1) espacio adicional.
    """
    primeros_errores = "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
    )
    logger.warning(
        "Cuerpo de request inválido en %s %s: %s", request.method, request.url.path, primeros_errores
    )
    body = _build_error_body(
        detail=f"El cuerpo de la solicitud no tiene el formato esperado: {primeros_errores}.",
        status=400,
        title="Bad Request",
        error_code="VF",
        error_label="Validación Fallida",
        instance=request.url.path,
        method=request.method,
    )
    return JSONResponse(status_code=400, content=body)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Red de seguridad final: captura cualquier excepción no anticipada
    que escape de un endpoint (ej. un error real de cómputo) y la
    traduce a un 500 en el formato exacto del enunciado, en vez de
    dejar escapar un traceback crudo al cliente.

    Args:
        request: request HTTP que originó el error.
        exc: excepción no manejada capturada.

    Returns:
        JSONResponse: error 500 en el formato exacto del enunciado.

    Complejidad:
        O(1) tiempo y espacio.
    """
    logger.error(
        "Excepción no manejada en %s %s: %r", request.method, request.url.path, exc, exc_info=True
    )
    body = _build_error_body(
        detail="Ocurrió un error interno al procesar la solicitud.",
        status=500,
        title="Internal Server Error",
        error_code="IE",
        error_label="Error Interno",
        instance=request.url.path,
        method=request.method,
    )
    return JSONResponse(status_code=500, content=body)
