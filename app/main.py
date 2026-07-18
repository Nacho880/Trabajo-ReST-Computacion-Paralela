"""
app/main.py
============

Punto de entrada de la API REST "Servicio ReST: Resumen estadístico"
para Cruz Morada — Universidad Tecnológica Metropolitana, Computación
Paralela y Distribuida.

Uso:
    uvicorn app.main:app --reload
    # Documentación interactiva (Swagger UI): http://127.0.0.1:8000/docs

Carga desatendida:
    Al iniciar, la aplicación carga automáticamente el CSV indicado por
    `config.CSV_PATH` (por defecto `data/ventas_completas.csv`,
    configurable con la variable de entorno `CPYD_CSV_PATH`), sin
    intervención manual, tal como exige el enunciado. Si la carga
    falla (archivo ausente, corrupto o con columnas faltantes), el
    proceso de arranque de Uvicorn falla explícitamente en vez de
    servir una API con datos parciales o inexistentes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.exceptions import (
    APIError,
    api_error_handler,
    request_validation_error_handler,
    unhandled_exception_handler,
)
from app.routers.ventas import router as ventas_router
from config import CSV_PATH, setup_logging
import src.db as db
from app.services.dataset import ingest_if_needed
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ciclo de vida de la aplicación: ejecuta la ingesta de dos pasadas de
    forma idempotente (si `etl_status` ya indica una carga completa
    anterior, no vuelve a procesar el CSV) y deja una conexión Postgres
    abierta en `app.state.conexion` para que la API consulte los datos
    directamente en cada request, sin mantener un DataFrame en memoria.

    Args:
        app: instancia de la aplicación FastAPI.

    Yields:
        None: control de vuelta a FastAPI mientras la aplicación sirve
        requests.
    """
    setup_logging()
    logger.info(
        "Iniciando API 'Servicio ReST: Resumen estadístico' (Cruz Morada). "
        "CSV=%s.", CSV_PATH,
    )
    # Flujo de arranque:
    #   1. ensure_schema_ready(): crea `ventas`/`etl_status` si no existen
    #      (CREATE TABLE IF NOT EXISTS, idempotente).
    #   2. ingest_if_needed(): si `etl_status` ya indica una carga
    #      completa anterior, no vuelve a leer el CSV; si no, ejecuta la
    #      ingesta de dos pasadas (también aplica ensure_schema
    #      internamente antes de decidir, por si la API se reinicia sin
    #      pasar por este paso 1).
    db.ensure_schema_ready()
    ingest_if_needed(CSV_PATH)
    app.state.conexion = db.get_connection()
    logger.info("API lista para servir /v1/estadisticas/ventas.")
    yield
    logger.info("Apagando la API.")
    conexion = getattr(app.state, "conexion", None)
    close = getattr(conexion, "close", None)
    if callable(close):
        close()


app = FastAPI(
    title="Servicio ReST: Resumen estadístico — Cruz Morada",
    description=(
        "API REST que entrega estadísticas de ventas (suma, conteo, "
        "promedio, mínimo, máximo, mediana y desviación estándar) para "
        "la cadena de farmacias Cruz Morada, con filtros combinables "
        "por género, edad, canal, producto, cliente, local y rango de "
        "fechas. Universidad Tecnológica Metropolitana — Computación "
        "Paralela y Distribuida."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ventas_router)

app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, request_validation_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
