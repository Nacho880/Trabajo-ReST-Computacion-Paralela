"""
src/utils/logger.py
====================

Utilidad mínima para obtener loggers de módulo consistentes en todo
el proyecto, evitando repetir `logging.getLogger(__name__)` con
configuración ad-hoc en cada archivo (principio DRY).

La configuración real de handlers (consola + archivo) vive en
`config.setup_logging()`, que debe llamarse una única vez (en este
proyecto, desde `app/main.py`) antes de usar cualquier logger de este
módulo.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger identificado por `name`, típicamente `__name__`
    del módulo que lo solicita.

    Args:
        name: nombre del logger (usualmente `__name__` del módulo llamador).

    Returns:
        logging.Logger: instancia de logger lista para usar. Hereda los
        handlers configurados en el logger raíz por `config.setup_logging()`.

    Complejidad:
        O(1) tiempo y espacio.
    """
    return logging.getLogger(name)
