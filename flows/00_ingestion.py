"""Ingesta: fuente remota -> capa Bronze.

Bronze conserva los valores tal como llegan. Los metadatos de ingesta van en
un archivo aparte (_ingestion_metadata.json); nunca se alteran los datos.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_LABELED_NAME,
    BRONZE_METADATA_NAME,
    BRONZE_QUALITY_NAME,
    BRONZE_UNLABELED_NAME,
)

log = get_logger(__name__)


def read_source(source: str) -> pd.DataFrame:
    """Lee la fuente completa (URL o ruta local) sin transformar valores."""
    return pd.read_csv(source)


def split_labeled(df: pd.DataFrame, n_labeled: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa los pasajeros con etiqueta (primeras n filas) de los que no la tienen."""
    labeled = df.iloc[:n_labeled].copy()
    unlabeled = df.iloc[n_labeled:].copy().reset_index(drop=True)
    return labeled, unlabeled


def quality_report(df: pd.DataFrame) -> dict:
    """Resumen de calidad: dimensiones, tipos y faltantes por columna.

    Lo que un analista mira a mano en el cuaderno, el pipeline lo deja
    registrado solo: así queda constancia de la calidad de CADA ingesta y se
    puede comparar una con otra.

    Devuelve un dict con las claves:
      n_rows, n_cols, dtypes ({col: str(dtype)}), nulls ({col: int}),
      null_pct ({col: float 0-100}).

    Los conteos se convierten con int() y float() a propósito: pandas devuelve
    tipos de NumPy (int64), que json.dumps no sabe serializar.
    """
    n_rows, n_cols = df.shape
    nulls = df.isnull().sum()
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "nulls": {str(col): int(n) for col, n in nulls.items()},
        # Si el DataFrame viniera vacío, n_rows es 0: se evita dividir por cero.
        "null_pct": {
            str(col): round(100.0 * int(n) / n_rows, 2) if n_rows else 0.0
            for col, n in nulls.items()
        },
    }


def run(source: str | None = None, n_labeled: int | None = None, out_dir: Path = BRONZE) -> dict:
    """Descarga, divide y guarda en Bronze. Devuelve los metadatos de la ingesta."""
    source = source or settings.local_path or settings.data_url
    n_labeled = n_labeled or settings.n_labeled

    log.info("Leyendo fuente: %s", source)
    df = read_source(source)
    labeled, unlabeled = split_labeled(df, n_labeled)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / BRONZE_LABELED_NAME
    unlabeled_path = out_dir / BRONZE_UNLABELED_NAME
    labeled.to_csv(labeled_path, index=False)
    unlabeled.to_csv(unlabeled_path, index=False)

    metadata = {
        "source": source,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "rows_labeled": len(labeled),
        "rows_unlabeled": len(unlabeled),
        "columns": list(df.columns),
        "sha256_labeled": sha256_of(labeled_path),
        "sha256_unlabeled": sha256_of(unlabeled_path),
    }
    (out_dir / BRONZE_METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
   
    # El reporte se calcula sobre 'labeled' y no sobre 'df' completo: es el
    # conjunto que alimentan el EDA y la capa Silver, y así los números del
    # archivo coinciden con los que muestra el cuaderno.
    report = quality_report(labeled)
    (out_dir / BRONZE_QUALITY_NAME).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Bronze listo: %d etiquetados, %d sin etiqueta", len(labeled), len(unlabeled))
    log.info(
        "Calidad: %d filas x %d columnas, %d columnas con faltantes",
        report["n_rows"],
        report["n_cols"],
        sum(1 for n in report["nulls"].values() if n > 0),
    )

    return metadata