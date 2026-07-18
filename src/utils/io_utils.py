"""
src/utils/io_utils.py
======================

Función genérica de entrada/salida para exportar resultados a JSON.

Centralizar el guardado aquí evita duplicar el manejo de errores de
escritura en cada módulo que necesite exportar un reporte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import TABLES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_json(data: dict[str, Any], filename: str) -> Path:
    """
    Guarda un diccionario como JSON en `output/tables/`.

    Args:
        data: diccionario serializable a JSON (se aplica `default=str`
            para tolerar tipos no nativos como `numpy.float64`).
        filename: nombre de archivo (ej. "reporte_limpieza_api.json").
            Si no incluye extensión, se asume `.json`.

    Returns:
        Path: ruta absoluta del archivo guardado.

    Raises:
        OSError: si no se puede escribir el archivo.

    Complejidad:
        O(n) tiempo y espacio, donde n = tamaño total de la estructura `data`.
    """
    if not filename.endswith(".json"):
        filename = f"{filename}.json"

    output_path = TABLES_DIR / filename
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info("JSON guardado en: %s", output_path)
    except OSError as exc:
        logger.error("No se pudo guardar el JSON '%s': %s", output_path, exc)
        raise

    return output_path
