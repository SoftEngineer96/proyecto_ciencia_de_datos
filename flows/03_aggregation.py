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
# Parte 9 — análisis crítico: interacciones, confusión y atípicos
# --------------------------------------------------------------------------
def _tabla_parquet(t: pd.DataFrame) -> pd.DataFrame:
    """Deja una tabla lista para Parquet: índice como columna y nombres de texto.

    Un pivot_table con columns='Pclass' produce columnas 1, 2, 3 (enteros), y
    Parquet exige que los nombres de columna sean texto.

    Además deja el eje de columnas SIN nombre. pivot_table lo bautiza con el
    nombre de la variable pivotada ('Sex', 'Pclass'), Parquet no guarda ese
    atributo, y al releer la tabla no sería idéntica a la escrita.
    """
    t = t.reset_index().rename(columns=str)
    t.columns.name = None
    return t


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 34: tasa de supervivencia con Pclass en filas y Sex en columnas.

    Es la interacción central del análisis: ninguna de las dos variables
    explica el resultado por sí sola.
    """
    return _tabla_parquet(
        df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean")
    )


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (1): tasa de supervivencia por puerto, SIN controlar por clase.

    Esta tabla es deliberadamente engañosa y existe para compararla con la
    siguiente: el puerto parece importar, pero lo que importa es la clase que
    embarcó en cada uno.
    """
    return _tabla_parquet(df.groupby("Embarked")["Survived"].agg(["mean", "count"]))


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (2): la misma tasa, ESTRATIFICADA por clase.

    Estratificar es comparar dentro de cada nivel de la tercera variable. Si el
    efecto del puerto se desvanece al mirar dentro de cada clase, el puerto no
    era la causa: era un indicador de la clase.
    """
    return _tabla_parquet(
        df.pivot_table(values="Survived", index="Embarked", columns="Pclass", aggfunc="mean")
    )


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (1): FareQ, el cuartil de tarifa calculado DENTRO de cada clase.

    Calcularlo sobre toda la tarifa no respondería la pregunta: los cuartiles
    globales serían casi un sinónimo de la clase. Dividiendo dentro de cada
    clase se compara a pasajeros de la misma clase entre sí, que es lo que
    permite ver si pagar más ayudaba por encima de la clase.

    duplicates='drop' evita el error cuando una clase tiene tarifas repetidas
    y dos bordes de cuartil coinciden.
    """
    out = df.copy()
    out["FareQ"] = out.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    return out


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (2): tasa por cuartil de tarifa, estratificada por clase."""
    con_cuartil = add_fare_quartile(df)
    return _tabla_parquet(
        con_cuartil.pivot_table(
            values="Survived", index="FareQ", columns="Pclass", aggfunc="mean"
        )
    )


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 37: tasa de supervivencia por tamaño de familia, con su n.

    El 'count' es imprescindible aquí: los grupos familiares grandes tienen
    muy pocos pasajeros y sus tasas son poco confiables.
    """
    return _tabla_parquet(df.groupby("Familia")["Survived"].agg(["mean", "count"]))


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Paso 38 (1): máscara booleana de atípicos por la regla del IQR.

    Es atípico lo que queda fuera de [Q1 − 1.5·IQR, Q3 + 1.5·IQR].

    Atípico NO significa erróneo. Una tarifa de 512 es real: alguien la pagó.
    La regla señala candidatos a revisar, no filas a eliminar.

    Los nulos no se marcan: una comparación con NaN da False.
    """
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 38 (2-3): cuántos atípicos hay por columna y cómo les fue.

    Para la edad usa AgeOriginal cuando existe: hacerlo sobre la Age imputada
    marcaría pasajeros normales como anómalos, porque la imputación encoge el
    IQR y estrecha el umbral (ver H17 en reports/conclusiones_eda.md).
    """
    columna_edad = "AgeOriginal" if "AgeOriginal" in df.columns else "Age"
    objetivo = {"Fare": "Fare", "Age": columna_edad, "Familia": "Familia"}

    filas = []
    for nombre, col in objetivo.items():
        if col not in df.columns:
            continue
        s = df[col]
        mask = outlier_mask_iqr(s)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        validos = int(s.notna().sum())

        fila = {
            "columna": nombre,
            "calculada_sobre": col,
            "n_atipicos": int(mask.sum()),
            "pct_atipicos": round(100.0 * int(mask.sum()) / validos, 2) if validos else 0.0,
            "limite_inferior": round(float(q1 - 1.5 * iqr), 2),
            "limite_superior": round(float(q3 + 1.5 * iqr), 2),
        }
        if "Survived" in df.columns:
            fila["tasa_atipicos"] = (
                round(float(df.loc[mask, "Survived"].mean()), 3) if mask.any() else float("nan")
            )
            fila["tasa_resto"] = round(float(df.loc[~mask, "Survived"].mean()), 3)
        filas.append(fila)

    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
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
