"""
app/services/estadisticas.py
==============================

Lógica de negocio del endpoint `/v1/estadisticas/ventas`: valida los 8
filtros del enunciado y calcula las 7 estadísticas exigidas.

Arquitectura de dos pasadas / Postgres:
    Con la API consultando Postgres directamente (en vez de un
    `pandas.DataFrame` completo en memoria), la validación de filtros se
    separa en dos capas:

        1. `construir_clausula_where(filtros)`: función PURA (sin base de
           datos) que valida cada filtro y lo traduce a un fragmento SQL
           parametrizado (`"columna = %s"`, ...) más sus parámetros. Aquí
           vive toda la lógica de negocio de los 8 filtros (mapeo de
           GENERO, enum de CANAL, UUID de ID_PERSONA, etc.). Se puede
           testear sin ninguna conexión a base de datos.
        2. `calcular_estadisticas(conexion, filtros)`: usa (1) para armar
           el `WHERE` y delega en `src.db.ejecutar_resumen` la consulta
           agregada real contra Postgres.

    `calcular_estadisticas_en_memoria(df, filtros)` opera sobre un
    `pandas.DataFrame` ya cargado en memoria, útil para explorar
    datasets pequeños fuera de la API (el router HTTP usa
    `calcular_estadisticas`, contra Postgres).
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd

import src.db as db
from config import API_STATS_TARGET_COLUMN
from app.exceptions import ValidacionFallidaError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Vocabularios de negocio (enums exigidos por el enunciado)
# ---------------------------------------------------------------------------
# GENERO en el CSV es Integer: el dataset codifica los 4 valores como
# 0=No especificado, 1=Masculino, 2=Femenino, 3=Otro. El filtro del
# enunciado recibe en cambio el texto ("Masculino", "Femenino", "Otro",
# "No especificado"), por lo que este diccionario traduce de uno al otro.
_GENERO_TEXTO_A_CODIGO: dict[str, int] = {
    "no especificado": 0,
    "masculino": 1,
    "femenino": 2,
    "otro": 3,
}
_GENERO_VALORES_PERMITIDOS = ("No especificado", "Masculino", "Femenino", "Otro")

_CANALES_VALIDOS: frozenset[str] = frozenset({"POS", "WEB", "APP", "CCT", "APR", "WPR"})

_NOMBRE_LEGIBLE: dict[str, str] = {
    "GENERO": "el género",
    "EDAD": "la edad",
    "CANAL": "el canal de venta",
    "CODIGO_PRODUCTO": "el código de producto",
    "ID_PERSONA": "el identificador de persona (UUID)",
    "LOCAL": "el ID de tienda",
    "FECHA_DESDE": "la fecha desde",
    "FECHA_HASTA": "la fecha hasta",
}

# Nombres de columna en Postgres (minúsculas; ver sql/ddl.sql), mapeados
# desde los nombres estándar del proyecto (mayúsculas).
_COLUMNA_SQL: dict[str, str] = {
    "GENERO": "genero",
    "EDAD": "edad",
    "CANAL": "canal",
    "SKU": "sku",
    "CODIGO_CLIENTE": "codigo_cliente",
    "LOCAL": "local",
    "FECHA": "fecha",
}


# ---------------------------------------------------------------------------
# Helpers de parseo (lanzan ValidacionFallidaError con mensaje exacto)
# ---------------------------------------------------------------------------
def _parsear_entero(valor: str, nombre_filtro: str) -> int:
    """Convierte `valor` a `int` o lanza `ValidacionFallidaError` con el mensaje del enunciado."""
    try:
        return int(str(valor).strip())
    except (TypeError, ValueError) as exc:
        raise ValidacionFallidaError(
            f"El valor '{valor}' no es un número entero válido para "
            f"{_NOMBRE_LEGIBLE[nombre_filtro]}."
        ) from exc


def _parsear_fecha_iso(valor: str, nombre_filtro: str) -> pd.Timestamp:
    """Convierte `valor` a `pandas.Timestamp`, exigiendo formato ISO-8601 estricto."""
    texto = str(valor).strip()
    try:
        return pd.Timestamp(datetime.fromisoformat(texto))
    except ValueError as exc:
        raise ValidacionFallidaError(
            f"El valor '{valor}' no es una fecha ISO-8601 válida para "
            f"{_NOMBRE_LEGIBLE[nombre_filtro]}."
        ) from exc


# ---------------------------------------------------------------------------
# Construcción de condiciones SQL por filtro: cada constructor devuelve
# (fragmento_sql, parametros) — nunca SQL con el valor interpolado
# directamente, siempre parametrizado.
# ---------------------------------------------------------------------------
def _condicion_genero(valor: str) -> tuple[str, list]:
    texto = str(valor).strip().lower()
    if texto not in _GENERO_TEXTO_A_CODIGO:
        raise ValidacionFallidaError(
            f"El valor '{valor}' no es un género válido. Los valores permitidos "
            f"son: {', '.join(_GENERO_VALORES_PERMITIDOS)}."
        )
    return f"{_COLUMNA_SQL['GENERO']} = %s", [_GENERO_TEXTO_A_CODIGO[texto]]


def _condicion_edad(valor: str) -> tuple[str, list]:
    edad = _parsear_entero(valor, "EDAD")
    return f"{_COLUMNA_SQL['EDAD']} = %s", [edad]


def _condicion_canal(valor: str) -> tuple[str, list]:
    texto = str(valor).strip().upper()
    if texto not in _CANALES_VALIDOS:
        raise ValidacionFallidaError(
            f"El valor '{valor}' no es un canal válido. Los valores "
            f"permitidos son: {', '.join(sorted(_CANALES_VALIDOS))}."
        )
    return f"{_COLUMNA_SQL['CANAL']} = %s", [texto]


def _condicion_codigo_producto(valor: str) -> tuple[str, list]:
    sku = _parsear_entero(valor, "CODIGO_PRODUCTO")
    return f"{_COLUMNA_SQL['SKU']} = %s", [sku]


def _condicion_id_persona(valor: str) -> tuple[str, list]:
    texto = str(valor).strip()
    try:
        uuid_normalizado = str(uuid.UUID(texto))
    except (ValueError, AttributeError) as exc:
        raise ValidacionFallidaError(
            f"El valor '{valor}' no es un UUID válido para "
            f"{_NOMBRE_LEGIBLE['ID_PERSONA']}."
        ) from exc
    return f"lower({_COLUMNA_SQL['CODIGO_CLIENTE']}) = %s", [uuid_normalizado]


def _condicion_local(valor: str) -> tuple[str, list]:
    local = _parsear_entero(valor, "LOCAL")
    return f"{_COLUMNA_SQL['LOCAL']} = %s", [local]


def _condicion_fecha_desde(valor: str) -> tuple[str, list]:
    fecha = _parsear_fecha_iso(valor, "FECHA_DESDE")
    return f"{_COLUMNA_SQL['FECHA']} >= %s", [fecha.to_pydatetime()]


def _condicion_fecha_hasta(valor: str) -> tuple[str, list]:
    fecha = _parsear_fecha_iso(valor, "FECHA_HASTA")
    return f"{_COLUMNA_SQL['FECHA']} <= %s", [fecha.to_pydatetime()]


_CONSTRUCTORES_CONDICION = {
    "GENERO": _condicion_genero,
    "EDAD": _condicion_edad,
    "CANAL": _condicion_canal,
    "CODIGO_PRODUCTO": _condicion_codigo_producto,
    "ID_PERSONA": _condicion_id_persona,
    "LOCAL": _condicion_local,
    "FECHA_DESDE": _condicion_fecha_desde,
    "FECHA_HASTA": _condicion_fecha_hasta,
}

FILTROS_PERMITIDOS: tuple[str, ...] = tuple(_CONSTRUCTORES_CONDICION.keys())


def construir_clausula_where(filtros: list[tuple[str, str]]) -> tuple[str, list]:
    """
    Valida `filtros` (0 o más, cualquier combinación) y los combina con
    AND lógico en un único fragmento SQL parametrizado.

    Función pura: no requiere conexión a base de datos, por lo que se
    puede testear de forma completamente aislada (ver
    `tests/test_estadisticas_service.py`).

    Args:
        filtros: lista de tuplas `(nombre_filtro, valor)`.

    Returns:
        tuple[str, list]: fragmento `WHERE` (sin la palabra `WHERE`, ya
        listo para anteponerle `AND` si hace falta) y la lista de
        parámetros en el mismo orden que los `%s` del fragmento. Cadena
        vacía y lista vacía si `filtros` está vacío.

    Raises:
        ValidacionFallidaError: si algún filtro no es uno de los 8
            permitidos, o su valor no es convertible al tipo esperado.
    """
    fragmentos: list[str] = []
    parametros: list = []

    for nombre_filtro, valor in filtros:
        nombre_filtro = nombre_filtro.strip().upper()
        constructor = _CONSTRUCTORES_CONDICION.get(nombre_filtro)
        if constructor is None:
            raise ValidacionFallidaError(
                f"El filtro '{nombre_filtro}' no es uno de los valores "
                f"permitidos. Los filtros permitidos son: "
                f"{', '.join(FILTROS_PERMITIDOS)}."
            )
        fragmento, params = constructor(valor)
        fragmentos.append(fragmento)
        parametros.extend(params)

    return " AND ".join(fragmentos), parametros


def calcular_estadisticas(conexion, filtros: list[tuple[str, str]]) -> dict:
    """
    Aplica `filtros` y calcula el resumen estadístico de
    `config.API_STATS_TARGET_COLUMN` consultando Postgres directamente
    (`src.db.ejecutar_resumen`), sin cargar ningún DataFrame en memoria.

    Args:
        conexion: conexión Postgres abierta (`src.db.get_connection()`).
        filtros: lista de tuplas `(nombre_filtro, valor)`.

    Returns:
        dict: resumen estadístico con las 7 claves del enunciado.

    Raises:
        ValidacionFallidaError: ver `construir_clausula_where`.
    """
    where_sql, parametros = construir_clausula_where(filtros)
    columna = _COLUMNA_SQL_OBJETIVO()
    resumen = db.ejecutar_resumen(conexion, columna, where_sql, parametros)

    logger.info(
        "Estadísticas calculadas sobre %d filtro(s): conteo=%d filas.",
        len(filtros), resumen["conteo"],
    )
    return resumen


def _COLUMNA_SQL_OBJETIVO() -> str:  # noqa: N802 - helper interno, no API pública
    return API_STATS_TARGET_COLUMN.lower()


# ---------------------------------------------------------------------------
# Cálculo en memoria (Pandas) sobre un DataFrame ya cargado, útil para
# análisis exploratorio de datasets pequeños fuera de la API HTTP.
# ---------------------------------------------------------------------------
def _mascara_genero(df: pd.DataFrame, valor: str) -> pd.Series:
    texto = str(valor).strip().lower()
    if texto in _GENERO_TEXTO_A_CODIGO:
        codigo = _GENERO_TEXTO_A_CODIGO[texto]
        return df["GENERO"] == codigo
    raise ValidacionFallidaError(
        f"El valor '{valor}' no es un género válido. Los valores permitidos "
        f"son: {', '.join(_GENERO_VALORES_PERMITIDOS)}."
    )


def _mascara_edad(df: pd.DataFrame, valor: str) -> pd.Series:
    edad = _parsear_entero(valor, "EDAD")
    return df["EDAD"] == edad


def _mascara_canal(df: pd.DataFrame, valor: str) -> pd.Series:
    texto = str(valor).strip().upper()
    if texto not in _CANALES_VALIDOS:
        raise ValidacionFallidaError(
            f"El valor '{valor}' no es un canal válido. Los valores "
            f"permitidos son: {', '.join(sorted(_CANALES_VALIDOS))}."
        )
    return df["CANAL"] == texto


def _mascara_codigo_producto(df: pd.DataFrame, valor: str) -> pd.Series:
    sku = _parsear_entero(valor, "CODIGO_PRODUCTO")
    return df["SKU"] == sku


def _mascara_id_persona(df: pd.DataFrame, valor: str) -> pd.Series:
    texto = str(valor).strip()
    try:
        uuid_normalizado = str(uuid.UUID(texto))
    except (ValueError, AttributeError) as exc:
        raise ValidacionFallidaError(
            f"El valor '{valor}' no es un UUID válido para "
            f"{_NOMBRE_LEGIBLE['ID_PERSONA']}."
        ) from exc
    return df["CODIGO_CLIENTE"].str.lower() == uuid_normalizado


def _mascara_local(df: pd.DataFrame, valor: str) -> pd.Series:
    local = _parsear_entero(valor, "LOCAL")
    return df["LOCAL"] == local


def _mascara_fecha_desde(df: pd.DataFrame, valor: str) -> pd.Series:
    fecha = _parsear_fecha_iso(valor, "FECHA_DESDE")
    return df["FECHA"] >= fecha


def _mascara_fecha_hasta(df: pd.DataFrame, valor: str) -> pd.Series:
    fecha = _parsear_fecha_iso(valor, "FECHA_HASTA")
    return df["FECHA"] <= fecha


_CONSTRUCTORES_MASCARA = {
    "GENERO": _mascara_genero,
    "EDAD": _mascara_edad,
    "CANAL": _mascara_canal,
    "CODIGO_PRODUCTO": _mascara_codigo_producto,
    "ID_PERSONA": _mascara_id_persona,
    "LOCAL": _mascara_local,
    "FECHA_DESDE": _mascara_fecha_desde,
    "FECHA_HASTA": _mascara_fecha_hasta,
}


def _resumen_estadistico(serie: pd.Series) -> dict:
    conteo = int(serie.shape[0])
    if conteo == 0:
        return {
            "suma": 0.0, "conteo": 0, "promedio": 0.0, "minimo": 0.0,
            "maximo": 0.0, "mediana": 0.0, "desviacion_estandar": 0.0,
        }

    suma = float(serie.sum())
    return {
        "suma": suma,
        "conteo": conteo,
        "promedio": suma / conteo,
        "minimo": float(serie.min()),
        "maximo": float(serie.max()),
        "mediana": float(serie.median()),
        "desviacion_estandar": float(serie.std(ddof=1)) if conteo > 1 else 0.0,
    }


def calcular_estadisticas_en_memoria(df: pd.DataFrame, filtros: list[tuple[str, str]]) -> dict:
    """
    Aplica `filtros` sobre `df` (`pandas.DataFrame` ya cargado en
    memoria) y calcula el resumen estadístico de
    `config.API_STATS_TARGET_COLUMN`.
    """
    mascara = pd.Series(True, index=df.index)

    for nombre_filtro, valor in filtros:
        nombre_filtro = nombre_filtro.strip().upper()
        constructor = _CONSTRUCTORES_MASCARA.get(nombre_filtro)
        if constructor is None:
            raise ValidacionFallidaError(
                f"El filtro '{nombre_filtro}' no es uno de los valores "
                f"permitidos. Los filtros permitidos son: "
                f"{', '.join(FILTROS_PERMITIDOS)}."
            )
        mascara = (mascara & constructor(df, valor)).fillna(False)

    mascara = mascara.astype(bool)
    serie_filtrada = (
        df.loc[mascara, API_STATS_TARGET_COLUMN].dropna().astype("float64")
    )
    return _resumen_estadistico(serie_filtrada)
