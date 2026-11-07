"""Modelado: Silver -> modelo entrenado y guardado.

Estado: en construcción. Se completa a lo largo de las Partes 10 a 17; cada
función pendiente indica su paso en docs/GUIA.md.

A diferencia de la capa Silver, aquí las funciones sin implementar lanzan
NotImplementedError en vez de devolver la entrada: no forman una cadena de
.pipe(), y un valor silencioso daría la impresión de que el modelo funciona.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from flows.common.config import settings
from flows.common.logging import get_logger
from flows.common.paths import SILVER, SILVER_NAME

log = get_logger(__name__)

TARGET = "Survived"
NUM_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Familia"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "TTitulo", "Alone"]
CABIN_NUM = ["HasCabin"]            # features "sospechosas": se evalúan en el Paso 61
CABIN_CAT = ["TCabin"]
CANDIDATE_COLUMNS = NUM_FEATURES + CAT_FEATURES + CABIN_NUM + CABIN_CAT
PRIMARY_METRIC = "roc_auc"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


# --------------------------------------------------------------------------
# Paso 39-40 — los datos que entran al modelo
# --------------------------------------------------------------------------
def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Lee Silver. Usa la edad ORIGINAL (con NaN): el modelo imputa por su cuenta.

    La Age de Silver ya fue imputada con el promedio de TODOS los pasajeros.
    Entrenar con ella filtraría información del conjunto de prueba hacia el de
    entrenamiento: el modelo vería, indirectamente, datos que debería
    desconocer. Por eso aquí se recupera AgeOriginal, con sus nulos, y la
    imputación se hace dentro del Pipeline (Parte 11), que la calcula solo con
    los datos de entrenamiento de cada partición.

    Las columnas excluidas y su motivo están en la auditoría del Paso 40:
    Survived y Survivor son el objetivo; PassengerId, Ticket y Name son
    identificadores que se memorizan sin generalizar; NSex, NEmbarked, NCabin
    y NTitulo duplican a sus categóricas imponiendo un orden que no existe, y
    se reemplazan por one-hot; Person es redundante con Age y Sex.
    """
    df = pd.read_parquet(Path(path) if path else SILVER / SILVER_NAME)
    X = df[CANDIDATE_COLUMNS].copy()
    X["Age"] = df["AgeOriginal"]
    return X, df[TARGET].astype(int)


# --------------------------------------------------------------------------
# Paso 41 — división entrenamiento / prueba
# --------------------------------------------------------------------------
def split(X: pd.DataFrame, y: pd.Series, test_size: float | None = None,
          random_state: int | None = None):
    """Devuelve X_train, X_test, y_train, y_test, estratificado por y.

    stratify=y mantiene la proporción de sobrevivientes en ambos conjuntos.
    Sin eso, con 891 filas y un 38 % de positivos, una partición desafortunada
    podría dejar el conjunto de prueba con una tasa muy distinta y hacer que
    las métricas no sean comparables.

    random_state fija la semilla: la misma división en cada ejecución, que es
    lo que permite comparar modelos entre sí y reproducir los resultados.

    Los valores por defecto vienen de settings (TEST_SIZE y RANDOM_STATE en
    .env), no de constantes escritas aquí: un cambio de configuración no
    debería exigir tocar el código.

    El conjunto de prueba se abre UNA SOLA VEZ, en la Parte 16. Todas las
    decisiones intermedias se toman con validación cruzada sobre el
    entrenamiento.
    """
    return train_test_split(
        X, y,
        test_size=settings.test_size if test_size is None else test_size,
        random_state=settings.random_state if random_state is None else random_state,
        stratify=y,
    )


# --------------------------------------------------------------------------
# Paso 42 — baselines
# --------------------------------------------------------------------------
def gender_rule_predict(X: pd.DataFrame) -> pd.Series:
    """Baseline 'regla de género': predice que sobrevive toda mujer.

    No es un modelo entrenado: es la regla que cualquiera deduciría del EDA en
    cinco minutos. Un modelo de machine learning que no la supere no aporta
    valor, por sofisticado que sea.
    """
    return (X["Sex"] == "female").astype(int)


def baseline_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compara los dos baselines del Paso 42 con las cuatro métricas de umbral.

    El baseline ingenuo ('siempre no sobrevive') existe para mostrar que la
    exactitud sola engaña: acierta ~62 % sin aprender nada, y su recall es 0.
    """
    predicciones = {
        "ingenuo (siempre 0)": pd.Series(0, index=X.index),
        "regla de genero": gender_rule_predict(X),
    }
    filas = []
    for nombre, pred in predicciones.items():
        filas.append({
            "baseline": nombre,
            "accuracy": round(accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Pendientes (Partes 11 a 17)
# --------------------------------------------------------------------------
def build_preprocessor():
    """Parte 11: ColumnTransformer con imputación, escalado y one-hot."""
    raise NotImplementedError("Parte 11")


def build_models():
    """Parte 12: diccionario de modelos candidatos dentro de su Pipeline."""
    raise NotImplementedError("Parte 12")


def evaluate_models(X, y):
    """Parte 13: validación cruzada estratificada de todos los candidatos."""
    raise NotImplementedError("Parte 13")


def tune_model(X, y):
    """Parte 14: búsqueda de hiperparámetros del modelo elegido."""
    raise NotImplementedError("Parte 14")


def final_report(X_test, y_test):
    """Parte 16: evaluación única sobre el conjunto de prueba."""
    raise NotImplementedError("Parte 16")
