"""
tests/test_csv_download.py
============================

Pruebas de `app/services/csv_download.py::download_csv_if_needed`, con
`requests.get` mockeado (no se hace ninguna llamada de red real).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

import app.services.csv_download as csv_download
from src.utils.validators import CSVDownloadError


class _RespuestaFalsa:
    """Simula una `requests.Response` en modo streaming."""

    def __init__(self, contenido: bytes, status_code: int = 200, content_length: int | None = None):
        self._contenido = contenido
        self.status_code = status_code
        self.headers = {"Content-Length": str(content_length or len(contenido))}

    def __enter__(self) -> "_RespuestaFalsa":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._contenido), chunk_size):
            yield self._contenido[i : i + chunk_size]


def test_no_descarga_si_el_archivo_ya_existe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "ventas.csv"
    csv_path.write_text("FECHA;CANAL\n2026-01-01;POS\n", encoding="utf-8")

    llamadas = []
    monkeypatch.setattr(csv_download.requests, "get", lambda *a, **k: llamadas.append(1))

    csv_download.download_csv_if_needed(csv_path)

    assert llamadas == []  # requests.get nunca se llamó
    assert csv_path.read_text(encoding="utf-8").startswith("FECHA")


def test_sin_url_configurada_lanza_error_claro(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "no-existe" / "ventas.csv"
    monkeypatch.setattr(csv_download, "CSV_DOWNLOAD_URL", "")

    with pytest.raises(CSVDownloadError, match="no hay una URL de descarga"):
        csv_download.download_csv_if_needed(csv_path)


def test_descarga_exitosa_crea_el_archivo_y_limpia_el_temporal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "subcarpeta" / "ventas.csv"
    contenido = b"FECHA;CANAL\n2026-01-01;POS\n" * 1000

    monkeypatch.setattr(csv_download, "CSV_DOWNLOAD_URL", "https://ejemplo.test/ventas.csv")
    monkeypatch.setattr(
        csv_download.requests, "get", lambda *a, **k: _RespuestaFalsa(contenido)
    )

    csv_download.download_csv_if_needed(csv_path)

    assert csv_path.exists()
    assert csv_path.read_bytes() == contenido
    # El archivo temporal no debe quedar tirado en el directorio.
    assert not (csv_path.with_name(csv_path.name + ".part")).exists()


def test_error_de_red_lanza_csv_download_error_y_no_deja_archivo_parcial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "ventas.csv"

    def _get_falla(*a, **k):
        raise requests.exceptions.ConnectionError("no hay red")

    monkeypatch.setattr(csv_download, "CSV_DOWNLOAD_URL", "https://ejemplo.test/ventas.csv")
    monkeypatch.setattr(csv_download.requests, "get", _get_falla)

    with pytest.raises(CSVDownloadError, match="No se pudo descargar"):
        csv_download.download_csv_if_needed(csv_path)

    assert not csv_path.exists()
    assert not (csv_path.with_name(csv_path.name + ".part")).exists()


def test_http_error_lanza_csv_download_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "ventas.csv"

    monkeypatch.setattr(csv_download, "CSV_DOWNLOAD_URL", "https://ejemplo.test/ventas.csv")
    monkeypatch.setattr(
        csv_download.requests, "get", lambda *a, **k: _RespuestaFalsa(b"", status_code=404)
    )

    with pytest.raises(CSVDownloadError, match="No se pudo descargar"):
        csv_download.download_csv_if_needed(csv_path)

    assert not csv_path.exists()


def test_descarga_vacia_lanza_csv_download_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "ventas.csv"

    monkeypatch.setattr(csv_download, "CSV_DOWNLOAD_URL", "https://ejemplo.test/ventas.csv")
    monkeypatch.setattr(csv_download.requests, "get", lambda *a, **k: _RespuestaFalsa(b""))

    with pytest.raises(CSVDownloadError, match="archivo vacío"):
        csv_download.download_csv_if_needed(csv_path)

    assert not csv_path.exists()
