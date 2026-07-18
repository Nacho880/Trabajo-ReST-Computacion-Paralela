"""
config.py
=========

Módulo de configuración global del proyecto.

Centraliza:
    - Definición de rutas de entrada/salida.
    - Esquema y formato del CSV de origen (separador, columnas, dtypes).
    - Configuración de logging (consola + archivo).
    - Ruta del CSV y columna objetivo que usa la API REST.

Diseño:
    Este módulo NO debe importar nada de `src/` para evitar dependencias
    circulares. Es la base de la que todos los demás módulos dependen.

Variables de entorno:
    Se cargan desde `.env` (raíz del proyecto) mediante `python-dotenv`
    si el archivo existe, y también desde el entorno del sistema
    operativo. Una variable ya exportada en el shell tiene prioridad
    sobre el valor de `.env` (comportamiento por defecto de
    `load_dotenv`, que no sobreescribe variables ya presentes en
    `os.environ`). Ver `.env.example` para la lista completa de
    variables soportadas.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# BASE_DIR apunta a la raíz del proyecto (directorio donde vive este archivo).
BASE_DIR: Path = Path(__file__).resolve().parent

# Carga las variables definidas en `.env` (si el archivo existe) al
# entorno del proceso, para que las lecturas de `os.environ.get(...)`
# más abajo las encuentren sin necesidad de exportarlas manualmente en
# la terminal antes de arrancar la API.
load_dotenv(BASE_DIR / ".env")

DATA_DIR: Path = BASE_DIR / "data"
OUTPUT_DIR: Path = BASE_DIR / "output"
TABLES_DIR: Path = OUTPUT_DIR / "tables"
LOGS_DIR: Path = OUTPUT_DIR / "logs"

# Se asegura la existencia de los directorios de salida en tiempo de import.
# Esto evita errores de "No such file or directory" al primer guardado.
for _directory in (OUTPUT_DIR, TABLES_DIR, LOGS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API REST (servicio ReST: resumen estadístico) — Cruz Morada
# ---------------------------------------------------------------------------
def get_csv_path() -> Path:
    """
    Obtiene la ruta del archivo CSV que la API carga automáticamente al
    iniciar (sin intervención manual, según exige el enunciado del
    servicio ReST).

    Lee la variable de entorno ``CPYD_CSV_PATH``. Si no está definida,
    usa ``data/ventas_completas.csv``, relativa a `BASE_DIR`
    si la ruta no es absoluta.

    No se hardcodea ninguna ruta: siempre se resuelve a través de esta
    función, tal como exige el enunciado ("No hardcodees rutas").

    Returns:
        Path: ruta absoluta al CSV de entrada de la API.

    Complejidad:
        O(1) tiempo y espacio.
    """
    raw_value = os.environ.get("CPYD_CSV_PATH", "data/ventas_completas.csv")
    path = Path(raw_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


CSV_PATH: Path = get_csv_path()

# URL pública desde donde se descarga `ventas_completas.csv` si no se
# encuentra en `CSV_PATH` al iniciar la API (ver
# `app/services/csv_download.py::download_csv_if_needed`). Debe ser un
# link de descarga DIRECTA (el archivo debe empezar a descargarse de
# inmediato al abrir la URL, sin páginas intermedias) — por ejemplo, un
# link de Dropbox con `dl=1` al final. Se puede sobreescribir con la
# variable de entorno `CPYD_CSV_DOWNLOAD_URL`, para no tener que editar
# el código si cambia. Vacío por defecto: si no se configura y el CSV
# local no existe, la API falla al iniciar con un error explícito en
# vez de arrancar sin datos.
CSV_DOWNLOAD_URL: str = os.environ.get("CPYD_CSV_DOWNLOAD_URL", "")

# Tiempo máximo (segundos) para establecer la conexión y para cada lectura
# de datos durante la descarga, antes de abortar con un error claro.
CSV_DOWNLOAD_TIMEOUT_SECONDS: int = 30

# Columna sobre la que la API calcula suma/conteo/promedio/mínimo/máximo/
# mediana/desviación estándar (monto de venta en CLP). Se centraliza aquí
# (no como literal repetido en `app/`) para que cambiarla sea un ajuste de
# una sola línea.
API_STATS_TARGET_COLUMN: str = "MONTO_APLICADO"


# ---------------------------------------------------------------------------
# Base de datos (PostgreSQL genérico) — arquitectura de dos pasadas
# ---------------------------------------------------------------------------
# Cadena de conexión a la base de datos de PRODUCCIÓN. Cualquier instancia
# PostgreSQL compatible sirve (local, Docker, on-premise o administrada);
# no se requiere ningún proveedor en particular. Vacía por defecto: si no
# está configurada, `src/db.py::get_connection()` falla explícitamente en
# vez de intentar conectar a "algo".
DATABASE_URL: str = os.environ.get("CPYD_DATABASE_URL", "")

VENTAS_TABLE: str = "ventas"
ETL_STATUS_TABLE: str = "etl_status"

# Si es False, `src/db.py::ensure_schema` omite los `CREATE INDEX`
# secundarios de `ventas` (fecha, genero, canal, local, sku,
# codigo_cliente). Para cargas extremadamente grandes, cada índice
# adicional puede duplicar/triplicar el espacio en disco requerido; en
# ese caso conviene crear el esquema sin índices, cargar los datos, y
# crear los índices manualmente después. Controlado por
# CPYD_CREATE_INDEXES (por defecto "true").
CREATE_INDEXES: bool = os.environ.get("CPYD_CREATE_INDEXES", "true").strip().lower() in (
    "1", "true", "yes", "si", "sí",
)

# Si es False, la ingesta omite el cálculo de FRECUENCIA_COMPRA (requiere
# un GROUP BY por CODIGO_CLIENTE sobre todo el dataset), la variable
# derivada más costosa del pipeline de feature engineering. Controlado
# por CPYD_ENABLE_FEATURE_ENGINEERING (por defecto "true"). Ver
# `src/feature_engineering.py` y la sección "Feature engineering
# opcional" del README.
ENABLE_FEATURE_ENGINEERING: bool = os.environ.get(
    "CPYD_ENABLE_FEATURE_ENGINEERING", "true"
).strip().lower() in ("1", "true", "yes", "si", "sí")

# Cadena de conexión a una base Postgres de PRUEBAS, usada únicamente por
# los tests de integración marcados `@pytest.mark.integration` (ver
# tests/conftest.py). Si no está definida, esos tests se saltan
# automáticamente: la suite completa corre en cualquier máquina sin
# necesidad de tener Postgres instalado.
TEST_DATABASE_URL: str = os.environ.get("CPYD_TEST_DATABASE_URL", "")

# Tamaño de partición objetivo para Dask durante la ingesta (Pasada 1 y
# Pasada 2). Se reutiliza el mismo valor para ambas pasadas: las
# estadísticas globales de la Pasada 1 deben aplicarse sobre el mismo
# particionado que procesa la Pasada 2.
INGEST_BLOCKSIZE: str = os.environ.get("CPYD_INGEST_BLOCKSIZE", "64MB")


# ---------------------------------------------------------------------------
# Formato del CSV de origen
# ---------------------------------------------------------------------------
# El delimitador real del archivo es punto y coma (no coma), un formato
# común en exportaciones de Excel en configuración regional
# latinoamericana/europea. Se centraliza aquí para que `data_loader.py` sea
# la única fuente de verdad y no queden "magic strings" de delimitador
# repetidos.
CSV_SEPARATOR: str = ";"
CSV_QUOTECHAR: str = '"'

RAW_EXPECTED_COLUMNS: tuple[str, ...] = (
    "FECHA",
    "CANAL",
    "SKU",
    "PRODUCTO",
    "UNIDADES",
    "PORCENTAJE DESCUENTO",
    "MONTO APLICADO",
    "BOLETA",
    "LOCAL",
    "CODIGO CLIENTE",
    "RUN CLIENTE",
    "NOMBRES",
    "APELLIDOS",
    "FECHA NACIMIENTO",
    "GENERO",
)

COLUMN_RENAME_MAP: dict[str, str] = {
    "FECHA": "FECHA",
    "CANAL": "CANAL",
    "SKU": "SKU",
    "PRODUCTO": "PRODUCTO",
    "UNIDADES": "UNIDADES",
    "PORCENTAJE DESCUENTO": "PORCENTAJE_DESCUENTO",
    "MONTO APLICADO": "MONTO_APLICADO",
    "BOLETA": "BOLETA",
    "LOCAL": "LOCAL",
    "CODIGO CLIENTE": "CODIGO_CLIENTE",
    "RUN CLIENTE": "RUN_CLIENTE",
    "NOMBRES": "NOMBRES",
    "APELLIDOS": "APELLIDOS",
    "FECHA NACIMIENTO": "FECHA_NACIMIENTO",
    "GENERO": "GENERO",
}

STANDARD_COLUMNS: tuple[str, ...] = tuple(COLUMN_RENAME_MAP.values())

EXPECTED_DTYPES: dict[str, str] = {
    # CANAL se lee como "string" (no "category") a propósito: ver
    # comentario en `data_loader.load_and_validate` sobre por qué
    # convertirla a categórica con `dtype="category"` directamente en
    # `dask.dataframe.read_csv` produce un orden de categorías NO
    # reproducible entre corridas.
    "CANAL": "string",
    "SKU": "Int64",
    "PRODUCTO": "string",
    "UNIDADES": "Int64",
    "PORCENTAJE_DESCUENTO": "float64",
    "MONTO_APLICADO": "float64",
    "BOLETA": "Int64",
    "LOCAL": "Int64",
    "CODIGO_CLIENTE": "string",
    "RUN_CLIENTE": "string",
    "NOMBRES": "string",
    "APELLIDOS": "string",
    "GENERO": "Int64",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class _ConsoleProgressFilter(logging.Filter):
    """
    Filtra qué registros llegan a la consola para mantenerla legible.

    Criterio: la consola solo muestra información "de alto nivel" (arranque
    de la API, carga del CSV) y cualquier WARNING/ERROR real que requiera
    atención. El detalle fino (conteos por columna, reportes de limpieza,
    etc.) sigue registrándose íntegro en el archivo de log (FileHandler),
    pero no satura la terminal.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name in ("__main__", "root", "app.main", "app.services.dataset")


def setup_logging(log_filename: str = "pipeline.log") -> logging.Logger:
    """
    Configura el logging global del proyecto (consola + archivo).

    Crea un logger raíz con dos handlers:
        1. StreamHandler -> salida por consola (nivel INFO).
        2. FileHandler   -> archivo en ``output/logs/<log_filename>``
           (nivel DEBUG, para trazabilidad completa).

    Idempotente: si se llama más de una vez, no duplica handlers.

    Args:
        log_filename: nombre del archivo de log dentro de ``output/logs/``.

    Returns:
        logging.Logger: logger raíz ya configurado.

    Complejidad:
        O(1) tiempo y espacio (configuración fija, independiente del volumen
        de datos que luego se registre).
    """
    root_logger = logging.getLogger()

    # Evita agregar handlers duplicados si setup_logging() se llama más de una vez
    # (por ejemplo, en tests que importan main varias veces).
    if getattr(root_logger, "_cpyd_configured", False):
        return root_logger

    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_ConsoleProgressFilter())

    log_path = LOGS_DIR / log_filename
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger._cpyd_configured = True  # marca idempotencia

    root_logger.info("Logging inicializado. log_file=%s", log_path)
    return root_logger
