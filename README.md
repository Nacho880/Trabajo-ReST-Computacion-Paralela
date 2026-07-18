# Servicio ReST: Resumen estadístico — Cruz Morada

Universidad Tecnológica Metropolitana — Computación Paralela y Distribuida.

API REST que entrega estadísticas de ventas (suma, conteo, promedio, mínimo,
máximo, mediana, desviación estándar) para Cruz Morada. Al iniciar, procesa
e inserta el CSV de ventas en PostgreSQL mediante particiones (sin
intervención manual y sin cargar el archivo completo en memoria en ningún
momento — ver "Procesamiento paralelo" más abajo). Los filtros (género,
edad, canal, producto, cliente, local, rango de fechas) se resuelven con
SQL parametrizado directamente contra la base de datos.

## Requisitos

- Python 3.12+
- PostgreSQL 14+ — cualquier instancia sirve (local, Docker, on-premise o
  administrada). **La base de datos no se incluye en el repositorio**: cada
  quien debe tener la suya y apuntarla en `CPYD_DATABASE_URL`.
- Espacio en disco suficiente para el CSV que se vaya a cargar.

## 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 2. Configurar la base de datos

Copiar `.env.example` a `.env` y completar `CPYD_DATABASE_URL` con la cadena
de conexión a su PostgreSQL:

```bash
cp .env.example .env
```

```env
CPYD_DATABASE_URL=postgresql://usuario:password@host:5432/nombre_bd
```

No hace falta crear tablas a mano: la API las crea solas al arrancar
(`sql/ddl.sql`, `CREATE TABLE IF NOT EXISTS`, idempotente). Si
`CPYD_DATABASE_URL` no está definida, la API falla al iniciar con un
mensaje explicando cómo configurarla.

## 3. Ubicar el CSV de ventas (opcional)

`.env.example` ya trae una `CPYD_CSV_DOWNLOAD_URL` con un link de descarga
directa: si no coloca el archivo manualmente, la API lo descarga sola en
el primer arranque. Si prefiere usar su propia copia local:

```bash
mkdir -p data
cp /ruta/a/su/ventas_completas.csv data/ventas_completas.csv
```

Por defecto se busca en `data/ventas_completas.csv` (configurable con
`CPYD_CSV_PATH`). Si el archivo local no existe, se intenta la descarga
desde `CPYD_CSV_DOWNLOAD_URL`; si esa variable está vacía y tampoco hay
archivo local, la API falla al iniciar con un error explícito.

## 4. Levantar la API

```bash
uvicorn app.main:app --reload
```

En el primer arranque, la API crea el esquema y procesa e inserta el CSV
completo en PostgreSQL mediante particiones (esto puede tardar según el
tamaño del archivo). En arranques posteriores detecta que la carga ya se
hizo y no la repite. Si el CSV no existe, está vacío/corrupto o le faltan
columnas, `uvicorn` **falla al iniciar** con un error explícito — nunca se
sirve la API con datos incompletos.

Documentación interactiva (Swagger UI): **http://127.0.0.1:8000/docs**

## Procesamiento paralelo

La carga del CSV se realiza con **Dask DataFrame**, que divide el archivo
en particiones independientes y permite operar sobre cada una en paralelo,
sin necesidad de tener el dataset completo en memoria (requisito del
enunciado: "implementar mecanismos de procesamiento paralelo... ej. Dask,
PySpark"). El flujo, implementado en `src/db.py` y
`app/services/dataset.py`, es de dos pasadas:

1. **Pasada 1:** Dask recorre el CSV de forma perezosa y calcula
   estadísticos globales (medianas, cotas de outliers, medias/desviaciones,
   frecuencia de compra por cliente) sin insertar nada todavía.
2. **Pasada 2:** se vuelve a recorrer el CSV partición por partición. Cada
   partición se limpia, se enriquece con las variables derivadas y se
   inserta en PostgreSQL mediante `COPY` masivo (`src/db.py::copy_dataframe`);
   luego se libera de memoria antes de procesar la siguiente.

## Arquitectura general

```
CSV
 │
 ▼
Dask DataFrame (particionado)
 │
 ▼
Procesamiento por partición (limpieza + variables derivadas)
 │
 ▼
PostgreSQL (COPY masivo, particionado por fecha)
 │
 ▼
FastAPI REST
 │
 ▼
Consultas GET / POST -> /v1/estadisticas/ventas
```

## Variables de entorno opcionales

| Variable | Por defecto | Para qué sirve |
|---|---|---|
| `CPYD_CSV_PATH` | `data/ventas_completas.csv` | Ruta del CSV de entrada |
| `CPYD_CSV_DOWNLOAD_URL` | (vacío) | URL de descarga directa si el CSV no existe localmente |
| `CPYD_DATABASE_URL` | (vacío, requerida) | Conexión a PostgreSQL |
| `CPYD_TEST_DATABASE_URL` | (vacío) | Conexión a una BD de pruebas, solo para tests de integración |
| `CPYD_CREATE_INDEXES` | `true` | Si `false`, omite los índices secundarios de `ventas` al crear el esquema (útil para cargas muy grandes, ver `sql/ddl.sql`) |
| `CPYD_ENABLE_FEATURE_ENGINEERING` | `true` | Si `false`, omite el cálculo de `frecuencia_compra` (el más costoso, requiere agrupar todo el dataset por cliente) |
| `CPYD_INGEST_BLOCKSIZE` | `64MB` | Tamaño de partición de Dask durante la carga |

## Endpoint

Ruta base: **`/v1/estadisticas/ventas`**.

- **`GET`**: permite consultar utilizando filtros mediante parámetros URL
  (query params), todos opcionales.
- **`POST`**: permite enviar filtros mediante un cuerpo JSON (`consultas`),
  también opcional.

Ambos calculan sobre la columna `MONTO_APLICADO` y devuelven el mismo
formato exitoso (200):

```json
{
  "suma": 1500.5,
  "conteo": 42,
  "promedio": 35.73,
  "minimo": 10.0,
  "maximo": 100.0,
  "mediana": 30.0,
  "desviacion_estandar": 25.4
}
```

Ejemplo ilustrativo con valores de una carga a mayor escala (sin filtros,
sobre todo el dataset):

```json
{
  "suma": 54239120.0,
  "conteo": 324877,
  "promedio": 166.95,
  "minimo": 100.0,
  "maximo": 90000.0,
  "mediana": 8000.0,
  "desviacion_estandar": 12000.0
}
```

Filtros disponibles (combinables, cualquier cantidad, unidos con AND):
`GENERO`, `EDAD`, `CANAL`, `CODIGO_PRODUCTO`, `ID_PERSONA`, `LOCAL`,
`FECHA_DESDE`, `FECHA_HASTA`.

```bash
# GET sin filtros
curl http://127.0.0.1:8000/v1/estadisticas/ventas

# GET con filtros combinados
curl "http://127.0.0.1:8000/v1/estadisticas/ventas?CANAL=POS&LOCAL=1999&FECHA_DESDE=2026-01-01T00:00:00&FECHA_HASTA=2026-03-31T23:59:59"

# POST con filtros personalizados
curl -X POST http://127.0.0.1:8000/v1/estadisticas/ventas \
  -H "Content-Type: application/json" \
  -d '{
        "consultas": [
          {"consulta": "GENERO", "valor": "Femenino"},
          {"consulta": "EDAD", "valor": "31"},
          {"consulta": "CANAL", "valor": "POS"}
        ]
      }'
```

Errores (`400` validación, `500` error interno) siguen el formato exacto
del enunciado, por ejemplo:

```json
{
  "detail": "El valor 'qwerqwer' no es un número entero válido para el ID de tienda",
  "instance": "/v1/estadisticas/ventas",
  "status": 400,
  "title": "Bad Request",
  "type": "https://developer.mozilla.org/es/docs/Web/HTTP/Reference/Status/400",
  "timestamp": "2026-06-30T20:44:49.201437000Z",
  "errorCode": "VF",
  "errorLabel": "Validación Fallida",
  "method": "POST"
}
```

## Columnas originales vs. calculadas

| Origen | Columnas |
|---|---|
| **Del CSV** (solo renombradas a `snake_case`) | `fecha`, `canal`, `sku`, `producto`, `unidades`, `porcentaje_descuento`, `monto_aplicado`, `boleta`, `local`, `codigo_cliente`, `run_cliente`, `nombres`, `apellidos`, `fecha_nacimiento`, `genero` |
| **Calculadas** por el pipeline | `monto_por_unidad` (`monto_aplicado / unidades`), `edad` (a partir de `fecha` y `fecha_nacimiento`), `frecuencia_compra` (cantidad de boletas únicas asociadas al cliente en todo el dataset: `COUNT(DISTINCT boleta) GROUP BY codigo_cliente` — NO es un promedio de días entre compras ni compras por período; omitible, ver tabla de variables de entorno) |

## Tests

Actualmente cuenta con 103 pruebas automatizadas: 85 puras (corren sin
Postgres) y 18 de integración (requieren `CPYD_TEST_DATABASE_URL`).

```bash
# Pruebas puras (sin Postgres) — corren en cualquier máquina
pytest tests/ -v -m "not integration"

# Suite completa, incluyendo integración con Postgres real
export CPYD_TEST_DATABASE_URL=postgresql://usuario:password@host:5432/nombre_bd_test
pytest tests/ -v
```

Las pruebas marcadas `@pytest.mark.integration` requieren
`CPYD_TEST_DATABASE_URL` y se saltan automáticamente si no está definida.

## Antes de entregar

- [ ] Subir el código a un repositorio **GitHub** e incluir a **`sebasalazar`**
  (el académico) como colaborador.
- [ ] Confirmar que `datos.json` esté en la raíz del repo.
- [ ] Confirmar que `.env` (con credenciales reales) **no** se suba (ver
  `.gitignore`); solo `.env.example` va al repositorio.

## Estructura del proyecto

```
├── app/
│   ├── main.py                # FastAPI app, arranque (esquema + ingesta + conexión), Swagger
│   ├── exceptions.py          # Formato de error exacto del enunciado
│   ├── routers/ventas.py      # GET y POST /v1/estadisticas/ventas
│   ├── services/
│   │   ├── dataset.py         # Orquesta la carga del CSV a Postgres
│   │   ├── csv_download.py    # Descarga automática si el CSV no existe localmente
│   │   └── estadisticas.py    # Valida los 8 filtros -> SQL parametrizado -> Postgres
│   └── models/schemas.py      # Pydantic: body POST y respuesta
├── src/
│   ├── data_loader.py         # Carga con Dask (chunking), validación de esquema
│   ├── data_cleaning.py       # Imputación de nulos, outliers
│   ├── feature_engineering.py # EDAD, MONTO_POR_UNIDAD, FRECUENCIA_COMPRA
│   ├── db.py                  # Conexión, esquema, particionado, COPY masivo, consultas SQL
│   └── utils/                 # Logger, validadores, exportación a JSON
├── sql/ddl.sql                # Esquema de `ventas` (particionada por fecha) + `etl_status`
├── tests/                     # Pruebas puras + integración (@pytest.mark.integration)
├── config.py                  # Rutas, esquema del CSV, conexión a la BD, logging
├── datos.json                 # Muestra de datos de prueba
└── requirements.txt
```

## Decisiones y supuestos

1. **Columna de las estadísticas:** `MONTO_APLICADO`.
2. **`GENERO`:** el CSV lo guarda como entero; el filtro recibe texto.
   Mapeo confirmado con el profesor (Sebastián Salazar Molina):
   `0="No especificado"`, `1="Masculino"`, `2="Femenino"`, `3="Otro"`.
3. **Filas con valor nulo en la columna de un filtro** se excluyen de ese
   filtro en vez de producir un error.
4. **Resultado vacío:** se devuelve `{"suma": 0.0, "conteo": 0, ...}` con
   todos los campos en 0, no un error ni `null`.
5. **Desviación estándar:** muestral (`ddof=1`).
