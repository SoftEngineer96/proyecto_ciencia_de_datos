"""Pruebas del modelado.

Se añaden a medida que se implementan las funciones de flows/05_ml_models.py.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

ml = importlib.import_module("flows.05_ml_models")


@pytest.fixture
def silver(tmp_path):
    """Silver mínima con todas las columnas que load_dataset necesita."""
    n = 120
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Survived": np.r_[np.zeros(74, dtype=int), np.ones(46, dtype=int)],
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": rng.choice(["male", "female"], n),
        "Age": np.full(n, 29.7),                       # ya imputada
        "AgeOriginal": np.r_[np.full(20, np.nan), rng.uniform(1, 70, n - 20)],
        "SibSp": rng.integers(0, 3, n),
        "Parch": rng.integers(0, 2, n),
        "Fare": rng.uniform(5, 100, n),
        "Familia": rng.integers(0, 4, n),
        "Embarked": rng.choice(["S", "C", "Q"], n),
        "TTitulo": rng.choice(["Mr", "Mrs", "Miss", "Otro"], n),
        "Alone": rng.choice(["Alone", "with Family"], n),
        "HasCabin": rng.integers(0, 2, n),
        "TCabin": rng.choice(["A", "C", "X"], n),
        "Survivor": "No",
        "PassengerId": np.arange(1, n + 1),
    })
    ruta = tmp_path / "titanic_silver.parquet"
    df.to_parquet(ruta, index=False)
    return ruta


# --- Paso 40: qué entra y qué no --------------------------------------------

def test_load_dataset_returns_x_and_y(silver):
    X, y = ml.load_dataset(silver)
    assert len(X) == len(y) == 120
    assert list(X.columns) == ml.CANDIDATE_COLUMNS


def test_load_dataset_uses_age_original_not_imputed(silver):
    """Entrenar con la Age imputada filtraría informacion del conjunto de prueba."""
    X, _ = ml.load_dataset(silver)
    assert X["Age"].isna().sum() == 20          # conserva los nulos reales
    assert not (X["Age"] == 29.7).all()         # no es la columna ya imputada


def test_load_dataset_excludes_leaking_and_identifier_columns(silver):
    """El objetivo, los identificadores y las codificaciones N* quedan fuera."""
    X, _ = ml.load_dataset(silver)
    prohibidas = {"Survived", "Survivor", "PassengerId", "Ticket", "Name",
                  "NSex", "NEmbarked", "NCabin", "NTitulo", "Person"}
    assert not prohibidas & set(X.columns)


def test_target_is_integer(silver):
    _, y = ml.load_dataset(silver)
    assert y.dtype.kind in "iu"
    assert set(y.unique()) <= {0, 1}


# --- Paso 41: división ------------------------------------------------------

def test_split_is_stratified(silver):
    """La tasa de supervivencia debe ser casi igual en ambos conjuntos."""
    X, y = ml.load_dataset(silver)
    _X_tr, _X_te, y_tr, y_te = ml.split(X, y)
    assert abs(y_tr.mean() - y_te.mean()) < 0.05
    assert abs(y.mean() - y_tr.mean()) < 0.05


def test_split_respects_test_size_and_loses_nothing(silver):
    X, y = ml.load_dataset(silver)
    X_tr, X_te, _y_tr, _y_te = ml.split(X, y, test_size=0.25)
    assert len(X_te) == 30
    assert len(X_tr) + len(X_te) == len(X)
    assert set(X_tr.index) & set(X_te.index) == set()     # sin solapamiento


def test_split_is_reproducible(silver):
    """La misma semilla da la misma division: sin eso no se pueden comparar modelos."""
    X, y = ml.load_dataset(silver)
    a = ml.split(X, y, random_state=7)[0].index.tolist()
    b = ml.split(X, y, random_state=7)[0].index.tolist()
    c = ml.split(X, y, random_state=8)[0].index.tolist()
    assert a == b
    assert a != c


def test_split_defaults_come_from_settings(silver):
    """Un cambio en .env no debe exigir tocar el codigo."""
    from flows.common.config import settings
    X, y = ml.load_dataset(silver)
    X_te = ml.split(X, y)[1]
    assert len(X_te) == round(len(X) * settings.test_size)


# --- Paso 42: baselines -----------------------------------------------------

def test_gender_rule_predicts_every_woman_survives(silver):
    X, _ = ml.load_dataset(silver)
    pred = ml.gender_rule_predict(X)
    assert (pred[X["Sex"] == "female"] == 1).all()
    assert (pred[X["Sex"] == "male"] == 0).all()


def test_baseline_scores_has_both_rows_and_four_metrics(silver):
    X, y = ml.load_dataset(silver)
    tabla = ml.baseline_scores(X, y).set_index("baseline")
    assert len(tabla) == 2
    assert list(tabla.columns) == ["accuracy", "precision", "recall", "f1"]


def test_naive_baseline_has_zero_recall(silver):
    """Prediciendo siempre 0 no encuentra ni un superviviente: la exactitud sola engana."""
    X, y = ml.load_dataset(silver)
    tabla = ml.baseline_scores(X, y).set_index("baseline")
    assert tabla.loc["ingenuo (siempre 0)", "recall"] == 0.0
    assert tabla.loc["ingenuo (siempre 0)", "accuracy"] == pytest.approx(1 - y.mean(), abs=1e-4)


# --- Pendientes -------------------------------------------------------------

def test_later_parts_fail_loudly():
    for fn in (ml.build_preprocessor, ml.build_models):
        with pytest.raises(NotImplementedError):
            fn()
