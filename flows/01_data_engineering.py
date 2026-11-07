"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().

Estado: esqueleto en construcción. Las funciones todavía sin implementar
devuelven el DataFrame sin tocarlo (funciones de paso), de modo que la cadena
de build_silver() funcione completa desde el primer día y 'make silver'
siempre produzca un archivo. Se van llenando una a una; cada TODO indica en
qué paso de docs/GUIA.md se implementa.

Cuidado: un cuerpo escrito como '...' devuelve None, y el primer .pipe() de la
cadena fallaría con AttributeError. Por eso las funciones pendientes llevan
'return df' explícito y no '...'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_LABELED_NAME,
    SILVER,
    SILVER_FINAL_NAME,
    SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]


# --------------------------------------------------------------------------
# Contrato de entrada
# --------------------------------------------------------------------------
def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Contrato de entrada: falla ruidosamente si Bronze no es lo esperado.

    Una capa es confiable solo si se niega a procesar datos que no cumplen su
    contrato. Es preferible un ValueError aquí que un resultado silenciosamente
    incorrecto veinte pasos después.

    Verifica dos cosas:
      1. Que estén todas las columnas de EXPECTED_COLUMNS.
      2. Que PassengerId no tenga duplicados (es la clave de cada pasajero).

    Devuelve el mismo df intacto para poder encadenarla con .pipe().
    """
    faltantes = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Bronze incumple el contrato: faltan columnas {faltantes}")

    duplicados = int(df["PassengerId"].duplicated().sum())
    if duplicados:
        raise ValueError(f"Bronze incumple el contrato: {duplicados} PassengerId duplicados")

    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Fija los tipos de las columnas numéricas.

    Survived llega como float (0.0 / 1.0) porque en la fuente completa hay
    filas sin etiqueta; en el conjunto etiquetado no hay nulos, así que se
    puede convertir a int sin perder nada.

    Age y Fare se dejan en float a propósito: Age tiene faltantes, y NaN solo
    existe en punto flotante.

    Trabaja sobre una copia: las funciones de esta capa son puras y no mutan
    lo que reciben.
    """
    out = df.copy()
    for col in ("Survived", "Pclass", "SibSp", "Parch"):
        out[col] = out[col].astype(int)
    for col in ("Age", "Fare"):
        out[col] = out[col].astype(float)
    return out


# --------------------------------------------------------------------------
# Transformaciones pendientes (ver docs/GUIA.md)
# --------------------------------------------------------------------------
def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 8: 'child' si Age < 16; si no, el valor de Sex.

    Regla explícita para edad desconocida: en NumPy y pandas, NaN < 16 es
    False, así que un pasajero sin edad NUNCA queda marcado como 'child' y
    conserva su sexo. Eso subestima a los niños sin edad registrada —el sesgo
    se cuantifica en el Reto 8.1—, pero aquí la regla está documentada y
    probada en lugar de ocurrir en silencio.

    Va antes de impute_age en build_silver a propósito: Person debe calcularse
    sobre la edad ORIGINAL, no sobre una imputada.
    """
    out = df.copy()
    out["Person"] = np.where(out["Age"] < 16, "child", out["Sex"])
    return out

def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pasos 11 y 24: TCabin (cubierta) y HasCabin (0/1).

    La primera letra de Cabin es la cubierta del barco ('C85' -> 'C'). Donde no
    hay cabina registrada se pone 'X': convierte el "no sé" en una categoría
    explícita en vez de un hueco, que es la opción (c) de las tres decisiones
    de limpieza de la Parte 5.

    HasCabin separa el HECHO de que el dato exista de su contenido. Importa
    porque el Paso 5 mostró que ese hecho no es aleatorio: falta en el 18.5 %
    de primera clase y en el 97.6 % de tercera, así que "no hay cabina
    registrada" es en sí información sobre el pasajero.

    Cabin original se conserva en Silver: nunca se descarta información en
    silencio.
    """
    out = df.copy()
    out["TCabin"] = out["Cabin"].str[0].fillna("X")
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    return out


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 14: Familia (SibSp + Parch) y Alone ('Alone' / 'with Family').

    SibSp cuenta hermanos y cónyuges a bordo; Parch, padres e hijos. Su suma es
    el total de familiares que viajaban con el pasajero, sin contarlo a él.
    Familia = 0 significa que viajaba solo.

    Se guardan las dos: Familia conserva el tamaño (un grupo de 7 no es lo
    mismo que uno de 1) y Alone da la versión categórica para los gráficos.
    """
    out = df.copy()
    out["Familia"] = out["SibSp"] + out["Parch"]
    out["Alone"] = np.where(out["Familia"] > 0, "with Family", "Alone")
    return out


def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 16: Survivor, etiqueta legible de Survived (0 -> 'No', 1 -> 'Yes').

    Es solo para rotular gráficos de conteo. Survived se conserva numérica
    porque su media ES la tasa de supervivencia: con valores 0 y 1,
    df['Survived'].mean() da directamente la proporción de supervivientes.
    Esa es la columna que usan los gráficos que promedian (catplot de puntos,
    lmplot); Survivor solo sirve para las etiquetas.
    """
    out = df.copy()
    out["Survivor"] = out["Survived"].map({0: "No", 1: "Yes"})
    return out

def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 25: TTitulo, extraído de Name."""
    return df  # TODO Paso 25


def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 26: agrupa títulos raros con TITLE_MAP y MAIN_TITLES; el resto, 'Otro'."""
    return df  # TODO Paso 26


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 21: imputa los 2 faltantes de Embarked."""
    return df  # TODO Paso 21


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 22: imputa Age conservando AgeOriginal para auditar."""
    return df  # TODO Paso 22


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Pasos 21, 23, 24 y 26: versiones numéricas NSex, NEmbarked, NCabin, NTitulo."""
    return df  # TODO Pasos 21, 23, 24, 26


# --------------------------------------------------------------------------
# Salida
# --------------------------------------------------------------------------
def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 32: COPIA sin Name, Sex, Embarked ni Cabin.

    Silver completo conserva todo para el análisis crítico de la Parte 9; esta
    vista reducida alimenta el modelado. Solo elimina las que existan, para que
    funcione mientras las columnas derivadas aún no se han creado.
    """
    originales = [c for c in ("Name", "Sex", "Embarked", "Cabin") if c in df.columns]
    return df.drop(columns=originales)


def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 33: contrato de salida. Sin nulos en las columnas derivadas.

    Todavía sin implementar: las columnas que debe vigilar (Person, Alone,
    TCabin, TTitulo, NSex, NEmbarked, NCabin, NTitulo) no existen hasta que se
    completen las transformaciones de arriba. AgeOriginal sí podrá tener nulos.
    """
    return df  # TODO Paso 33


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME
