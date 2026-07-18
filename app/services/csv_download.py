"""
app/services/csv_download.py
==============================

Mecanismo de obtención automática del CSV de ventas: si el archivo no
existe localmente en `config.CSV_PATH`, se descarga desde
`config.CSV_DOWNLOAD_URL` antes de que el pipeline de carga
(`src.data_loader.load_and_validate`) intente leerlo.

Este módulo es puramente de "obtención del archivo" — no toca carga,
limpieza, feature engineering, ni ninguna otra lógica existente. Se
mantiene separado de `app/services/dataset.py` (responsabilidad única):
uno consigue el archivo, el otro orquesta su procesamiento.

Diseño:
    - Descarga en streaming (``requests.get(..., stream=True)``) y por
      bloques: nunca se carga el archivo completo en memoria antes de
      escribirlo a disco, indispensable para un CSV de cientos de MB.
    - Escritura atómica: se descarga primero a un archivo temporal
      (``<nombre>.part``) y solo se renombra al nombre final
      (``os.replace``, atómico a nivel de sistema operativo) si la
      descarga se completó sin errores. Así, si el proceso se interrumpe
      a mitad de la descarga, jamás queda un `ventas_completas.csv`
      parcial/corrupto que un reinicio posterior confundiría con un
      archivo válido ya presente.
    - Cualquier fallo (sin URL configurada, error de red, HTTP 4xx/5xx,
      timeout, archivo vacío) se traduce en `CSVDownloadError`, que
      detiene el arranque de la API (mismo comportamiento que los demás
      errores de carga en `src.utils.validators`) — nunca se sirve la
      API sin datos.
"""

from __future__ import annotations

from pathlib import Path

import requests

from config import CSV_DOWNLOAD_TIMEOUT_SECONDS, CSV_DOWNLOAD_URL
from src.utils.logger import get_logger
from src.utils.validators import CSVDownloadError

logger = get_logger(__name__)

_TAMANO_BLOQUE_BYTES = 1024 * 1024  # 1 MB por bloque de descarga.


def download_csv_if_needed(csv_path: str | Path) -> None:
    """
    Garantiza que `csv_path` exista localmente antes de que el pipeline
    de carga lo lea, descargándolo desde `config.CSV_DOWNLOAD_URL` si
    hace falta.

    Args:
        csv_path: ruta local esperada del CSV (típicamente
            `config.CSV_PATH`).

    Raises:
        CSVDownloadError: si el archivo no existe localmente Y no se
            pudo descargar (sin URL configurada, error de red, HTTP de
            error, timeout, o archivo descargado vacío).

    Complejidad:
        O(1) si el archivo ya existe. Si se descarga: O(n) tiempo y
        espacio en disco, n = tamaño del archivo remoto; O(1) memoria
        adicional (streaming por bloques, no se materializa el archivo
        completo en RAM).
    """
    csv_path = Path(csv_path)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        logger.info("CSV encontrado en '%s'. Se utilizará el archivo local.", csv_path)
        return

    if not CSV_DOWNLOAD_URL:
        raise CSVDownloadError(
            f"El archivo '{csv_path}' no existe y no hay una URL de descarga "
            "configurada. Defina config.CSV_DOWNLOAD_URL (o la variable de "
            "entorno CPYD_CSV_DOWNLOAD_URL) con un link de descarga directa, "
            "o coloque el CSV manualmente en esa ruta."
        )

    logger.info(
        "CSV no encontrado en '%s'. Descargando desde %s ...", csv_path, CSV_DOWNLOAD_URL
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _descargar_a_archivo_temporal(CSV_DOWNLOAD_URL, csv_path)
    logger.info(
        "Descarga completada correctamente: '%s' (%.1f MB). Iniciando carga del dataset...",
        csv_path, csv_path.stat().st_size / (1024 ** 2),
    )


def _descargar_a_archivo_temporal(url: str, destino: Path) -> None:
    """
    Descarga `url` por bloques a un archivo temporal junto a `destino`
    y lo renombra atómicamente al finalizar sin errores.

    Args:
        url: URL de descarga directa.
        destino: ruta final donde debe quedar el archivo descargado.

    Raises:
        CSVDownloadError: ante cualquier error de red, HTTP, timeout, o
            si el archivo resultante queda vacío.

    Complejidad:
        O(n) tiempo y espacio en disco, n = tamaño del archivo remoto;
        O(1) memoria adicional.
    """
    temporal = destino.with_name(destino.name + ".part")

    try:
        with requests.get(url, stream=True, timeout=CSV_DOWNLOAD_TIMEOUT_SECONDS) as respuesta:
            respuesta.raise_for_status()
            tamano_total = int(respuesta.headers.get("Content-Length", 0))
            descargado = 0
            ultimo_porcentaje_logueado = -10

            with temporal.open("wb") as archivo_temporal:
                for bloque in respuesta.iter_content(chunk_size=_TAMANO_BLOQUE_BYTES):
                    if not bloque:
                        continue
                    archivo_temporal.write(bloque)
                    descargado += len(bloque)

                    if tamano_total > 0:
                        porcentaje = int(descargado * 100 / tamano_total)
                        if porcentaje >= ultimo_porcentaje_logueado + 10:
                            logger.info(
                                "Descargando CSV: %d%% (%.1f/%.1f MB)",
                                porcentaje, descargado / (1024 ** 2), tamano_total / (1024 ** 2),
                            )
                            ultimo_porcentaje_logueado = porcentaje

    except requests.exceptions.RequestException as exc:
        temporal.unlink(missing_ok=True)
        raise CSVDownloadError(
            f"No se pudo descargar el CSV desde '{url}': {exc}"
        ) from exc

    if not temporal.exists() or temporal.stat().st_size == 0:
        temporal.unlink(missing_ok=True)
        raise CSVDownloadError(
            f"La descarga desde '{url}' finalizó pero produjo un archivo vacío."
        )

    temporal.replace(destino)
