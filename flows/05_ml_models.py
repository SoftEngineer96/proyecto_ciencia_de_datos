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
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    train_test_split,
)
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
# Paso 48 — validación cruzada de todos los candidatos
# --------------------------------------------------------------------------
def _scorers() -> dict:
    """Los scorers de SCORING, con zero_division=0 donde hace falta.

    El baseline 'dummy' predice siempre la clase mayoritaria, así que nunca
    emite un positivo y la precisión queda indefinida (0/0). Con los nombres
    de texto, scikit-learn llena la salida de UndefinedMetricWarning; con
    make_scorer se fija el valor en 0, que es la lectura correcta: un modelo
    que no predice positivos no acierta ninguno.
    """
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }


def make_cv(random_state=None, cv_folds=None) -> StratifiedKFold:
    """El particionador que usa TODO el proyecto, con los mismos parámetros.

    Que el Paso 48 y el Paso 51 compartan esta función no es un detalle de
    estilo: si los pliegues no fueran idénticos, las probabilidades fuera de
    muestra de la Parte 13 no corresponderían a la tabla de la Parte 12, y las
    dos lecturas no serían comparables.
    """
    semilla = settings.random_state if random_state is None else random_state
    pliegues = settings.cv_folds if cv_folds is None else cv_folds
    return StratifiedKFold(n_splits=pliegues, shuffle=True, random_state=semilla)


def cross_validate_candidates(X, y, num_features=None, cat_features=None,
                              random_state=None, cv_folds=None,
                              return_folds: bool = False):
    """Paso 48: validación cruzada estratificada de todos los candidatos.

    Un modelo puede memorizar el entrenamiento y fallar con datos nuevos. La
    validación cruzada de k pliegues divide el entrenamiento en k partes,
    entrena con k−1 y evalúa en la restante, rotando. La MEDIA de los k
    puntajes estima el desempeño y la DESVIACIÓN indica la estabilidad.

    Estratificada: cada pliegue conserva la proporción de supervivientes.
    shuffle=True con semilla fija: el orden original del archivo no sesga la
    partición, y el resultado es reproducible.

    Devuelve la tabla ordenada por la métrica principal. Con return_folds=True
    devuelve además el detalle por pliegue, en formato largo, para el boxplot
    del Paso 49.

    La columna 'brecha_roc_auc' es entrenamiento menos validación: cuánto peor
    le va al modelo con datos que no vio. Una brecha grande indica
    sobreajuste.

    Con 891 pasajeros la varianza es considerable: una diferencia de medio
    punto entre dos modelos no es evidencia de que uno sea mejor.
    """
    semilla = settings.random_state if random_state is None else random_state
    cv = make_cv(semilla, cv_folds)
    scorers = _scorers()

    filas, detalle = [], []
    for nombre, modelo in get_candidates(semilla).items():
        pipe = build_pipeline(modelo, num_features, cat_features)
        res = cross_validate(pipe, X, y, cv=cv, scoring=scorers,
                             return_train_score=True, n_jobs=1)

        fila = {"modelo": nombre}
        for metrica in SCORING:
            fila[f"{metrica}_mean"] = round(float(res[f"test_{metrica}"].mean()), 4)
            fila[f"{metrica}_std"] = round(float(res[f"test_{metrica}"].std()), 4)
        fila[f"train_{PRIMARY_METRIC}_mean"] = round(
            float(res[f"train_{PRIMARY_METRIC}"].mean()), 4
        )
        fila[f"brecha_{PRIMARY_METRIC}"] = round(
            float(res[f"train_{PRIMARY_METRIC}"].mean() - res[f"test_{PRIMARY_METRIC}"].mean()), 4
        )
        fila["fit_time_mean"] = round(float(res["fit_time"].mean()), 4)
        filas.append(fila)

        for metrica in SCORING:
            for i, valor in enumerate(res[f"test_{metrica}"]):
                detalle.append({"modelo": nombre, "metrica": metrica,
                                "pliegue": i, "valor": float(valor)})

        log.info(
            "%-7s %s=%.4f (+/-%.4f)  brecha=%.4f",
            nombre, PRIMARY_METRIC,
            fila[f"{PRIMARY_METRIC}_mean"], fila[f"{PRIMARY_METRIC}_std"],
            fila[f"brecha_{PRIMARY_METRIC}"],
        )

    tabla = (pd.DataFrame(filas)
             .sort_values(f"{PRIMARY_METRIC}_mean", ascending=False)
             .reset_index(drop=True))
    return (tabla, pd.DataFrame(detalle)) if return_folds else tabla


# --------------------------------------------------------------------------
# Parte 13 — evaluación en profundidad (Pasos 51 a 55)
# --------------------------------------------------------------------------
FINALISTS = ["hgb", "logreg", "forest"]     # decisión del Paso 50


def oof_proba(pipe, X, y, random_state=None, cv_folds=None) -> np.ndarray:
    """Paso 51: probabilidad de supervivencia fuera de muestra, por pasajero.

    cross_val_predict reparte a los pasajeros en los mismos k pliegues y
    predice cada uno con el modelo entrenado SIN él. El resultado es un vector
    con una probabilidad honesta por pasajero: ninguna la emitió un modelo que
    ya lo había visto.

    Esto permite mirar matriz de confusión, curvas, calibración y errores por
    subgrupo sin abrir el conjunto de prueba.

    Advertencia: el vector mezcla k modelos distintos, uno por pliegue. Sirve
    para diagnosticar el PROCEDIMIENTO, no para auditar un modelo concreto.
    """
    cv = make_cv(random_state, cv_folds)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def oof_proba_finalists(X, y, names=None, num_features=None, cat_features=None,
                        random_state=None, cv_folds=None) -> pd.DataFrame:
    """Una columna de probabilidades fuera de muestra por finalista."""
    semilla = settings.random_state if random_state is None else random_state
    candidatos = get_candidates(semilla)
    nombres = list(FINALISTS if names is None else names)
    return pd.DataFrame(
        {
            n: oof_proba(build_pipeline(candidatos[n], num_features, cat_features),
                         X, y, semilla, cv_folds)
            for n in nombres
        },
        index=pd.RangeIndex(len(y)),
    )


def confusion_table(y, pred) -> pd.DataFrame:
    """Paso 51: la matriz de confusión con los nombres del problema.

    Las etiquetas importan: '47 falsos positivos' no dice nada, '47 pasajeros
    a los que el modelo dio por supervivientes y no lo fueron' sí, porque ese
    es el error caro declarado en la sección 2 de reports/problema_ml.md.
    """
    m = confusion_matrix(y, pred, labels=[0, 1])
    return pd.DataFrame(
        m,
        index=["real: no sobrevivió", "real: sobrevivió"],
        columns=["predicho: no sobrevivió", "predicho: sobrevivió"],
    )


def threshold_sweep(y, proba) -> pd.DataFrame:
    """Paso 53: precisión, recall y F1 en cada umbral que cambia algo.

    precision_recall_curve devuelve un punto más que umbrales (el extremo
    donde no se predice ningún positivo); se descarta para que las cuatro
    columnas tengan el mismo largo.
    """
    precision, recall, umbral = precision_recall_curve(y, proba)
    precision, recall = precision[:-1], recall[:-1]
    suma = precision + recall
    f1 = np.divide(2 * precision * recall, suma,
                   out=np.zeros_like(precision, dtype=float), where=suma > 0)
    return pd.DataFrame({"umbral": umbral, "precision": precision,
                         "recall": recall, "f1": f1})


def best_threshold(y, proba, criterio: str = "f1") -> dict:
    """Paso 53: el umbral que maximiza el criterio pedido.

    0.5 es una convención, no una decisión. Este es el umbral que el criterio
    prefiere; cuál adoptar depende del costo de cada error, no de la métrica.
    """
    barrido = threshold_sweep(y, proba)
    fila = barrido.loc[barrido[criterio].idxmax()]
    return {c: round(float(fila[c]), 4) for c in barrido.columns}


def threshold_for_precision(y, proba, min_precision: float = 0.80) -> dict:
    """Paso 53: el umbral con más recall entre los que alcanzan esa precisión.

    La sección 2 de reports/problema_ml.md declaró que el falso positivo es el
    error caro para la aseguradora. Maximizar la precisión sin más llevaría el
    umbral a predecir casi nada; lo operativo es fijar un piso de precisión y,
    dentro de lo que lo cumple, conservar el mayor recall posible.

    'alcanzado' es False cuando ningún umbral llega al piso pedido: entonces
    se devuelve el de mayor precisión disponible, que no cumple el requisito.
    """
    barrido = threshold_sweep(y, proba)
    viables = barrido[barrido["precision"] >= min_precision]
    alcanzado = not viables.empty
    fila = (viables.loc[viables["recall"].idxmax()] if alcanzado
            else barrido.loc[barrido["precision"].idxmax()])
    salida = {c: round(float(fila[c]), 4) for c in barrido.columns}
    salida["alcanzado"] = alcanzado
    return salida


def calibration_table(y, proba, n_bins: int = 8) -> pd.DataFrame:
    """Paso 53: ¿la probabilidad dice la verdad?

    Un modelo está calibrado si, entre los pasajeros a los que asigna 0.7,
    sobrevivió cerca del 70 %. La tabla compara, por tramo de probabilidad, lo
    que el modelo prometió (proba_media) con lo que ocurrió (tasa_real).

    Se calcula a mano en vez de con calibration_curve porque esa función
    descarta los tramos vacíos y no devuelve el tamaño de cada uno: sin 'n' no
    se sabe si una desviación del 20 % son 90 pasajeros o 3.
    """
    datos = pd.DataFrame({"real": np.asarray(y, dtype=float),
                          "proba": np.asarray(proba, dtype=float)})
    cortes = np.linspace(0.0, 1.0, n_bins + 1)
    datos["tramo"] = pd.cut(datos["proba"], bins=cortes, include_lowest=True)
    tabla = (datos.groupby("tramo", observed=False)
             .agg(n=("real", "size"), proba_media=("proba", "mean"),
                  tasa_real=("real", "mean"))
             .reset_index())
    tabla["tramo"] = tabla["tramo"].astype(str)
    tabla["desvio"] = (tabla["tasa_real"] - tabla["proba_media"]).round(4)
    return tabla.round({"proba_media": 4, "tasa_real": 4})


def brier(y, proba) -> float:
    """Paso 53: error cuadrático medio de la probabilidad.

    0 es perfecto. Un modelo que siempre responde la tasa base (0.384) obtiene
    0.237, así que ese es el valor a batir, no 0.25.
    """
    return round(float(brier_score_loss(y, proba)), 4)


def subgroup_report(X, y, pred, by: str) -> pd.DataFrame:
    """Paso 54: exactitud, precisión y recall dentro de cada subgrupo.

    Una métrica global promedia sobre grupos que el modelo trata muy distinto.
    La columna 'n' viene siempre: una exactitud del 100 % sobre 6 pasajeros no
    es un resultado.
    """
    datos = pd.DataFrame({
        "grupo": pd.Series(X[by]).to_numpy(),
        "real": np.asarray(y),
        "pred": np.asarray(pred),
    })
    filas = [
        {
            "variable": by,
            "grupo": valor,
            "n": len(g),
            "tasa_real": round(float(g["real"].mean()), 4),
            "exactitud": round(float(accuracy_score(g["real"], g["pred"])), 4),
            "precision": round(float(precision_score(g["real"], g["pred"],
                                                     zero_division=0)), 4),
            "recall": round(float(recall_score(g["real"], g["pred"],
                                               zero_division=0)), 4),
        }
        for valor, g in datos.groupby("grupo", observed=False)
    ]
    return (pd.DataFrame(filas)
            .sort_values("exactitud")
            .reset_index(drop=True))


def confident_errors(X, y, proba, threshold: float = 0.5, n: int = 10) -> pd.DataFrame:
    """Paso 54: los errores en los que el modelo estaba más seguro.

    Un error con probabilidad 0.51 es ruido. Uno con 0.95 significa que el
    modelo tiene una creencia firme y equivocada, y eso suele señalar una
    variable que falta.
    """
    p = np.asarray(proba, dtype=float)
    salida = X.reset_index(drop=True).copy()
    salida["real"] = np.asarray(y)
    salida["proba"] = p.round(4)
    salida["pred"] = (p >= threshold).astype(int)
    salida["confianza"] = np.abs(p - threshold).round(4)
    return (salida[salida["pred"] != salida["real"]]
            .sort_values("confianza", ascending=False)
            .head(n)
            .reset_index(drop=True))


def learning_curve_table(pipe, X, y, train_sizes=None,
                         random_state=None, cv_folds=None) -> pd.DataFrame:
    """Paso 55: ¿conviene conseguir más datos o cambiar de modelo?

    Dos diagnósticos opuestos:
    - Las dos curvas convergen en un valor BAJO -> sesgo alto: el modelo es
      demasiado simple o faltan variables. Más datos no arreglan nada.
    - Queda una BRECHA grande que se cierra al crecer n -> varianza alta: más
      datos o más regularización sí ayudan.
    """
    cv = make_cv(random_state, cv_folds)
    tamanos = np.linspace(0.2, 1.0, 5) if train_sizes is None else train_sizes
    sizes, tr, va = learning_curve(pipe, X, y, cv=cv, train_sizes=tamanos,
                                   scoring=PRIMARY_METRIC, n_jobs=1)
    return pd.DataFrame({
        "n_entrenamiento": sizes,
        "train_mean": tr.mean(axis=1).round(4),
        "train_std": tr.std(axis=1).round(4),
        "val_mean": va.mean(axis=1).round(4),
        "val_std": va.std(axis=1).round(4),
        "brecha": (tr.mean(axis=1) - va.mean(axis=1)).round(4),
    })


# --------------------------------------------------------------------------
# Parte 14 — ajuste de hiperparámetros (Pasos 56 a 58)
# --------------------------------------------------------------------------
def get_param_grids() -> dict[str, dict]:
    """Paso 56: rejillas por modelo, con el prefijo 'model__' del Pipeline.

    El prefijo no es decorativo: la búsqueda recorre el pipeline COMPLETO, así
    que el preprocesamiento se reajusta en cada pliegue de cada combinación.
    Sin eso, la búsqueda tendría la misma fuga que el Paso 43 demostró.

    Las rejillas apuntan a lo que diagnosticó la curva de aprendizaje del Paso
    55: varianza alta. De ahí que 'forest' y 'hgb' muevan sobre todo los
    diales que limitan la capacidad de memorizar —profundidad, hojas mínimas,
    penalización L2— y no los que la aumentan.
    """
    return {
        # El enunciado propone además 'model__penalty': ['l2']. Se omite a
        # propósito: scikit-learn 1.8 deprecó ese argumento y lo retira en la
        # 1.10, así que llena la salida de FutureWarning. Como la rejilla
        # tenía un único valor y 'l2' ya es el comportamiento por defecto, el
        # espacio de búsqueda es idéntico sin él.
        "logreg": {
            "model__C": [0.01, 0.1, 1, 10],
        },
        "forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "hgb": {
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__max_iter": [100, 200],
            "model__l2_regularization": [0.0, 1.0],
        },
    }


def tune_model(X, y, name: str = "hgb", num_features=None, cat_features=None,
               random_state=None, cv_folds=None, n_jobs: int = -1) -> GridSearchCV:
    """Paso 57: búsqueda exhaustiva sobre la rejilla del modelo.

    Devuelve el GridSearchCV ya ajustado: trae best_params_, best_score_ y
    cv_results_ con las combinaciones probadas.

    best_score_ es OPTIMISTA. Es el máximo de muchos puntajes ruidosos, y el
    máximo de una muestra ruidosa está sesgado hacia arriba aunque ninguna
    combinación sea realmente mejor. El Paso 58 existe para medir ese sesgo.
    """
    semilla = settings.random_state if random_state is None else random_state
    cv = make_cv(semilla, cv_folds)
    pipe = build_pipeline(get_candidates(semilla)[name], num_features, cat_features)
    busqueda = GridSearchCV(pipe, get_param_grids()[name], scoring=PRIMARY_METRIC,
                            cv=cv, n_jobs=n_jobs, refit=True)
    busqueda.fit(X, y)
    log.info("%-7s mejor %s=%.4f  %s", name, PRIMARY_METRIC,
             busqueda.best_score_, busqueda.best_params_)
    return busqueda


def search_results(busqueda: GridSearchCV, top: int = 10) -> pd.DataFrame:
    """Paso 57: las mejores combinaciones y qué tan plana es la superficie.

    'dif_vs_mejor' responde la pregunta que importa: si las primeras filas
    están separadas por milésimas, el ajuste no aporta y conviene quedarse con
    la configuración más simple.
    """
    tabla = (pd.DataFrame(busqueda.cv_results_)
             [["rank_test_score", "mean_test_score", "std_test_score",
               "mean_fit_time", "params"]]
             .sort_values("rank_test_score")
             .head(top)
             .reset_index(drop=True))
    tabla["dif_vs_mejor"] = tabla["mean_test_score"].max() - tabla["mean_test_score"]
    tabla["params"] = tabla["params"].astype(str)
    return tabla.round({"mean_test_score": 4, "std_test_score": 4,
                        "mean_fit_time": 4, "dif_vs_mejor": 4})


def paired_fold_comparison(X, y, name: str = "hgb", best_params: dict | None = None,
                           num_features=None, cat_features=None,
                           random_state=None, cv_folds=None) -> pd.DataFrame:
    """Paso 58 (opción B): por defecto contra ajustado, en los MISMOS pliegues.

    Comparar dos medias obtenidas en particiones distintas confunde la mejora
    con la suerte del reparto. Usando los mismos pliegues, la diferencia
    pliegue a pliegue cancela esa fuente de ruido.

    Sesgo que hay que declarar: los hiperparámetros se eligieron mirando estos
    mismos pliegues, así que la diferencia que sale aquí es un TECHO, no una
    estimación. La cifra honesta es la de nested_cv_score.
    """
    semilla = settings.random_state if random_state is None else random_state
    cv = make_cv(semilla, cv_folds)
    por_defecto = build_pipeline(get_candidates(semilla)[name],
                                 num_features, cat_features)
    ajustado = build_pipeline(get_candidates(semilla)[name],
                              num_features, cat_features)
    if best_params:
        ajustado.set_params(**best_params)

    a = cross_val_score(por_defecto, X, y, cv=cv, scoring=PRIMARY_METRIC)
    b = cross_val_score(ajustado, X, y, cv=cv, scoring=PRIMARY_METRIC)
    return pd.DataFrame({
        "pliegue": range(len(a)),
        "por_defecto": a.round(4),
        "ajustado": b.round(4),
        "diferencia": (b - a).round(4),
    })


def nested_cv_score(X, y, name: str = "hgb", num_features=None, cat_features=None,
                    random_state=None, cv_folds=None, n_jobs: int = -1) -> dict:
    """Paso 58 (opción A): validación cruzada anidada.

    Mide el desempeño del PROCEDIMIENTO COMPLETO 'buscar hiperparámetros y
    entrenar', no el del mejor punto encontrado. En cada pliegue externo la
    búsqueda se rehace desde cero con los pliegues internos, y se evalúa sobre
    datos que esa búsqueda nunca vio.

    Es cara —pliegues externos x combinaciones x pliegues internos ajustes— y
    es la única forma de saber cuánto de best_score_ era sesgo de selección.

    Los pliegues externos usan otra semilla, para que no coincidan con los
    internos.
    """
    semilla = settings.random_state if random_state is None else random_state
    busqueda = GridSearchCV(
        build_pipeline(get_candidates(semilla)[name], num_features, cat_features),
        get_param_grids()[name], scoring=PRIMARY_METRIC,
        cv=make_cv(semilla, cv_folds), n_jobs=n_jobs, refit=True,
    )
    puntajes = cross_val_score(busqueda, X, y, scoring=PRIMARY_METRIC,
                               cv=make_cv(semilla + 1, cv_folds), n_jobs=1)
    return {"puntajes": [round(float(p), 4) for p in puntajes],
            "media": round(float(puntajes.mean()), 4),
            "desviacion": round(float(puntajes.std()), 4)}


# --------------------------------------------------------------------------
# Pendientes (Partes 15 a 17)
# --------------------------------------------------------------------------


def final_report(X_test, y_test):
    """Parte 16: evaluación única sobre el conjunto de prueba."""
    raise NotImplementedError("Parte 16")
