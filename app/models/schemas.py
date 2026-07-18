"""
app/models/schemas.py
======================

Esquemas Pydantic del contrato HTTP de `/v1/estadisticas/ventas`.

Se limitan estrictamente a la forma de la solicitud POST y de la
respuesta exitosa definidas en el enunciado, sin agregar campos
adicionales.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConsultaFiltro(BaseModel):
    """
    Un filtro individual dentro del body de `POST /v1/estadisticas/ventas`.

    Corresponde exactamente a `{"consulta": "GENERO", "valor": "Femenino"}`
    del enunciado. La validación de que `consulta` sea uno de los 8
    filtros permitidos y de que `valor` sea convertible al tipo esperado
    por ese filtro ocurre en `app/services/estadisticas.py`, no aquí:
    este modelo solo garantiza la FORMA del objeto (dos campos string),
    no su contenido de negocio.

    Attributes:
        consulta: nombre del filtro (ej. "GENERO", "EDAD").
        valor: valor del filtro, siempre como texto (tal como en el
            ejemplo del enunciado, incluso para filtros numéricos como
            EDAD: `"valor": "31"`).
    """

    consulta: str = Field(..., description="Nombre del filtro (ej. GENERO, EDAD, CANAL).")
    valor: str = Field(..., description="Valor del filtro, como texto (ej. '31', 'Femenino').")


class ConsultasRequest(BaseModel):
    """
    Body de `POST /v1/estadisticas/ventas`.

    `consultas` es opcional. Su tratamiento depende de CÓMO llega
    (Opción A, ver `app/routers/ventas.py::obtener_estadisticas_post`
    para la discusión completa de por qué se resuelve así):
        - Campo ausente (body `{}` o sin body): "sin filtros" (200),
          equivalente a un GET sin query params. Es la forma legítima
          de pedir "las consultas sin usar estos filtros" que exige el
          enunciado.
        - `null` EXPLÍCITO, o `[]` EXPLÍCITO: error 400 "Validación
          Fallida", tal como exige el enunciado ("consultas vacío o
          nulo"). Se distingue de "ausente" en `app/routers/ventas.py`
          usando `model_fields_set` (Pydantic solo lo incluye si el
          cliente mandó la clave `consultas`, sea cual sea su valor).

    Esta clase solo modela la FORMA del body; el chequeo de
    `null`/`[]` explícitos vive en el router (no aquí) porque requiere
    lanzar `ValidacionFallidaError`, una excepción de dominio HTTP que
    este módulo de esquemas deliberadamente no importa.

    Attributes:
        consultas: lista de filtros a aplicar (0 o más), en cualquier
            combinación.
    """

    consultas: list[ConsultaFiltro] | None = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "consultas": [
                        {"consulta": "LOCAL", "valor": "1999"},
                        {"consulta": "GENERO", "valor": "Femenino"},
                        {"consulta": "CANAL", "valor": "WEB"},
                    ]
                },
                {},
            ]
        }
    }

    def filtros_como_lista(self) -> list[tuple[str, str]]:
        """
        Normaliza `consultas` a una lista de tuplas `(nombre, valor)`,
        con `nombre` ya en mayúsculas (los filtros del enunciado son
        todos en mayúsculas: GENERO, EDAD, CANAL, etc.).

        Returns:
            list[tuple[str, str]]: lista vacía si `consultas` es `None`.

        Complejidad:
            O(k) tiempo y espacio, k = número de filtros.
        """
        if not self.consultas:
            return []
        return [(item.consulta.strip().upper(), item.valor) for item in self.consultas]


class EstadisticaResponse(BaseModel):
    """
    Respuesta exitosa de `GET`/`POST /v1/estadisticas/ventas`.

    Nombres de campo exactos exigidos por el enunciado (no se agregan
    campos adicionales, ej. no se incluye qué columna o filtros
    generaron el resultado).

    Attributes:
        suma: suma de la columna objetivo sobre las filas filtradas.
        conteo: número de filas filtradas (con valor no nulo en la
            columna objetivo).
        promedio: suma / conteo (0.0 si conteo es 0).
        minimo: valor mínimo.
        maximo: valor máximo.
        mediana: valor central (promedio de los dos centrales si
            conteo es par).
        desviacion_estandar: desviación estándar muestral (ddof=1).
    """

    suma: float
    conteo: int
    promedio: float
    minimo: float
    maximo: float
    mediana: float
    desviacion_estandar: float
