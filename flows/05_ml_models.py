"""Modelado: Silver -> modelo entrenado y guardado.

Estado: en construcción. Se completa a lo largo de las Partes 10 a 17; cada
función pendiente indica su paso en docs/GUIA.md.

A diferencia de la capa Silver, aquí las funciones sin implementar lanzan
NotImplementedError en vez de devolver la entrada: no forman una cadena de
.pipe(), y un valor silencioso daría la impresión de que el modelo funciona.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
# Paso 44 — imputación de edad por grupo, como transformador
# --------------------------------------------------------------------------
class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Rellena 'Age' con el promedio de su grupo, aprendido SOLO en fit().

    Es la imputación del Paso 22 convertida en transformador de scikit-learn,
    y esa conversión es lo que elimina la fuga: fit() calcula los promedios
    con los datos que recibe —dentro de la validación cruzada, solo los de
    entrenamiento de ese pliegue— y transform() los aplica a cualquier dato.

    Si en transform aparece un grupo que fit nunca vio, usa el promedio global
    del entrenamiento en lugar de dejar un nulo.
    """

    def __init__(self, group_cols=("Sex",), target="Age"):
        self.group_cols = group_cols
        self.target = target

    def fit(self, X, y=None):
        cols = list(self.group_cols)
        self.group_means_ = X.groupby(cols)[self.target].mean()
        self.global_mean_ = X[self.target].mean()
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.group_cols)
        means = self.group_means_.rename("_fill").reset_index()
        fill = X[cols].merge(means, on=cols, how="left")["_fill"].to_numpy(dtype=float)
        fill = np.where(np.isnan(fill), self.global_mean_, fill)
        X[self.target] = X[self.target].fillna(pd.Series(fill, index=X.index))
        return X


# --------------------------------------------------------------------------
# Paso 45 — el pipeline completo
# --------------------------------------------------------------------------
def build_pipeline(model, num_features=None, cat_features=None) -> Pipeline:
    """GroupAgeImputer -> ColumnTransformer -> modelo, todo en un Pipeline.

    Un Pipeline encadena los pasos en UN solo objeto con fit y predict. Al
    hacer fit, cada paso aprende únicamente de los datos que recibe —medianas,
    categorías, medias—; al predecir, aplica lo aprendido. Eso garantiza el
    mismo preprocesamiento en entrenamiento, validación y producción, sin
    duplicar código y sin que una estadística del conjunto de prueba se filtre
    al de entrenamiento.

    Numéricas: mediana por si quedara algún faltante (Fare, por ejemplo) y
    estandarización, que los modelos basados en distancia necesitan.

    Categóricas: la más frecuente para los faltantes, y one-hot en vez de un
    número por categoría. One-hot no impone un orden inexistente: evita que el
    modelo interprete que 'Q' (2) es mayor que 'S' (0).

    min_frequency=10 agrupa las categorías muy raras —como la cubierta T, con
    un solo pasajero— en lugar de crear columnas casi vacías que el modelo
    podría memorizar.

    sparse_threshold=0 fuerza salida densa, que todos los modelos aceptan.

    La imputación de edad agrupa por Sex Y Pclass, más específica que la del
    Paso 22, que solo usaba Sex. La comparación entre ambas está en el
    cuaderno 04.
    """
    num_features = NUM_FEATURES if num_features is None else list(num_features)
    cat_features = CAT_FEATURES if cat_features is None else list(cat_features)

    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
    ])
    prep = ColumnTransformer(
        [("num", num, num_features), ("cat", cat, cat_features)],
        sparse_threshold=0,
    )
    return Pipeline([
        ("age_imputer", GroupAgeImputer(group_cols=("Sex", "Pclass"))),
        ("prep", prep),
        ("model", model),
    ])


# --------------------------------------------------------------------------
# Paso 47 — modelos candidatos
# --------------------------------------------------------------------------
def get_candidates(random_state: int = 42) -> dict:
    """Diccionario {nombre: estimador sin entrenar}.

    Una familia variada: un baseline que no aprende nada, un modelo lineal
    interpretable, un árbol simple, dos ensambles y un modelo basado en
    distancia. Todos con la semilla fija, para que la comparación sea
    reproducible.

    'dummy' está incluido a propósito: si un modelo no lo supera con holgura,
    el problema está en los datos o en el preprocesamiento, no en el modelo.
    """
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000),
        "tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        "forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
    }


# --------------------------------------------------------------------------
# Pendientes (Partes 12 a 17)
# --------------------------------------------------------------------------
def cross_validate_candidates(X, y, num_features=None, cat_features=None,
                              random_state=None, cv_folds=None):
    """Paso 48: validación cruzada estratificada de todos los candidatos."""
    raise NotImplementedError("Paso 48")


def tune_model(X, y):
    """Parte 14: búsqueda de hiperparámetros del modelo elegido."""
    raise NotImplementedError("Parte 14")


def final_report(X_test, y_test):
    """Parte 16: evaluación única sobre el conjunto de prueba."""
    raise NotImplementedError("Parte 16")
