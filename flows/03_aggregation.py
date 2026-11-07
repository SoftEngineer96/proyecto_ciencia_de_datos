"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe.

Estado: en construcción. Las funciones de la Parte 9 todavía no están
implementadas y por eso no entran en el diccionario de run(): así 'make gold'
produce solo tablas reales en lugar de archivos vacíos. Cada una se va
añadiendo a TABLAS a medida que se implementa.

Gotcha de Parquet: exige que los nombres de columna sean texto, y guarda el
índice aparte. Por eso todas las tablas salen con .reset_index() y, cuando un
pivot produce columnas numéricas, con .rename(columns=str).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Paso 28 — estadísticas descriptivas por grupos
# --------------------------------------------------------------------------
def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cuatro resúmenes por grupo, listos para un informe.

    En las tres tablas de supervivencia se piden juntos 'mean', 'sum' y
    'count' a propósito: la tasa sola engaña. Un 100 % sobre 19 pasajeros y
    un 100 % sobre 2 no son el mismo hallazgo, y solo 'count' lo distingue.
    """
    por_grupo = ["mean", "sum", "count"]
    return {
        "age_by_survival": df.groupby("Survived")["Age"].describe().reset_index(),
        "survival_by_sex": df.groupby("Sex")["Survived"].agg(por_grupo).reset_index(),
        "survival_by_pclass": df.groupby("Pclass")["Survived"].agg(por_grupo).reset_index(),
        "survival_by_alone": df.groupby("Alone")["Survived"].agg(por_grupo).reset_index(),
    }


# --------------------------------------------------------------------------
# Paso 29 — prueba de independencia chi-cuadrado
# --------------------------------------------------------------------------
def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Prueba de independencia entre Sex y Survived.

    H0: las dos variables son independientes (el sexo no se asocia con
    sobrevivir). H1: hay asociación. Con p < 0.05 se rechaza H0.

    Lo que el p-valor NO dice: ni que la asociación sea grande, ni que sea
    causal. Solo que es poco probable que una diferencia así surja del azar
    si H0 fuera cierta. El tamaño del efecto se lee en las tasas de
    survival_by_sex, no aquí.

    Los valores se convierten a tipos nativos porque json.dumps no serializa
    los de NumPy.
    """
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p, gl, _esperados = stats.chi2_contingency(tabla)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(gl)}


# --------------------------------------------------------------------------
# Pendientes de la Parte 9 (Pasos 34 a 38)
# --------------------------------------------------------------------------
def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 34: tasa de supervivencia con Pclass en filas y Sex en columnas."""
    raise NotImplementedError("Paso 34")


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (1): tasa de supervivencia por puerto de embarque."""
    raise NotImplementedError("Paso 35")


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (2): la misma tasa, estratificada por clase."""
    raise NotImplementedError("Paso 35")


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (1): agrega el cuartil de tarifa."""
    raise NotImplementedError("Paso 36")


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (2): tasa por cuartil de tarifa, estratificada por clase."""
    raise NotImplementedError("Paso 36")


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 37: tasa de supervivencia por tamaño de familia."""
    raise NotImplementedError("Paso 37")


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Paso 38 (1): máscara de atípicos por el criterio del rango intercuartílico."""
    raise NotImplementedError("Paso 38")


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 38 (2-3): resumen de atípicos por columna."""
    raise NotImplementedError("Paso 38")


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Solo las tablas ya implementadas. Las de la Parte 9 se añaden aquí
    # conforme se completen, en lugar de escribir parquets vacíos ahora.
    tables = {
        **group_stats(df),
        # "survival_by_class_sex": survival_by_class_sex(df),          # Paso 34
        # "survival_by_embarked": survival_by_embarked(df),            # Paso 35
        # "survival_by_embarked_class": survival_by_embarked_class(df),  # Paso 35
        # "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),  # Paso 36
        # "survival_by_family": survival_by_family(df),                # Paso 37
        # "outlier_summary": outlier_summary(df),                      # Paso 38
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)

    prueba = chi2_sex_survival(df)
    (out_dir / "chi2_sex_survival.json").write_text(
        json.dumps(prueba, indent=2), encoding="utf-8"
    )

    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    log.info(
        "Chi-cuadrado Sex x Survived: chi2=%.2f, p=%.3g, gl=%d",
        prueba["chi2"], prueba["p_value"], prueba["dof"],
    )
    return out_dir
