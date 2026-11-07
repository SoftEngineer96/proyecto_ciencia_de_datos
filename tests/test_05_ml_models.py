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


# --- Parte 11: pruebas del enunciado ---------------------------------------

@pytest.fixture
def toy():
    """Fixture del enunciado: sintético, para probar sin depender de Silver."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({
        "Age": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 70, n)),
        "SibSp": rng.integers(0, 4, n),
        "Parch": rng.integers(0, 3, n),
        "Fare": rng.uniform(5, 100, n),
        "Familia": rng.integers(0, 6, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": rng.choice(["male", "female"], n),
        "Embarked": rng.choice(["S", "C", "Q"], n),
        "TTitulo": rng.choice(["Mr", "Mrs", "Miss", "Otro"], n),
        "Alone": rng.choice(["Alone", "with Family"], n),
        "HasCabin": rng.integers(0, 2, n),
        "TCabin": rng.choice(list("XXXXCB"), n),
    })
    y = pd.Series(((X["Sex"] == "female") & (rng.random(n) < 0.9)).astype(int))
    return X, y


def test_group_age_imputer_uses_only_training_statistics():
    """Prueba del enunciado: el promedio sale de TRAIN, no de los datos a transformar."""
    train = pd.DataFrame({"Sex": ["male", "male", "female"], "Age": [10.0, 20.0, 40.0]})
    test = pd.DataFrame({"Sex": ["male", "other"], "Age": [np.nan, np.nan]})
    out = ml.GroupAgeImputer(group_cols=("Sex",)).fit(train).transform(test)
    assert out.loc[0, "Age"] == 15.0                          # promedio de hombres en TRAIN
    assert out.loc[1, "Age"] == pytest.approx(70 / 3)         # grupo no visto: promedio global


def test_pipeline_fits_and_predicts_probabilities(toy):
    """Prueba del enunciado."""
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((0 <= proba) & (proba <= 1)).all()


def test_split_is_stratified_and_reproducible(toy):
    """Prueba del enunciado."""
    X, y = toy
    a = ml.split(X, y, test_size=0.25, random_state=1)
    b = ml.split(X, y, test_size=0.25, random_state=1)
    pd.testing.assert_index_equal(a[0].index, b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.03


def test_candidates_include_baseline_and_are_seeded():
    """Prueba del enunciado."""
    cands = ml.get_candidates(42)
    assert "dummy" in cands and "logreg" in cands and len(cands) >= 5


# --- Parte 11: pruebas complementarias -------------------------------------

def test_group_age_imputer_does_not_mutate_input():
    train = pd.DataFrame({"Sex": ["male", "female"], "Age": [10.0, 40.0]})
    test = pd.DataFrame({"Sex": ["male"], "Age": [np.nan]})
    ml.GroupAgeImputer(group_cols=("Sex",)).fit(train).transform(test)
    assert test["Age"].isna().all()


def test_group_age_imputer_leaves_known_ages_untouched():
    """Solo rellena nulos: una edad real no se toca."""
    train = pd.DataFrame({"Sex": ["male", "male"], "Age": [10.0, 20.0]})
    test = pd.DataFrame({"Sex": ["male", "male"], "Age": [np.nan, 55.0]})
    out = ml.GroupAgeImputer(group_cols=("Sex",)).fit(train).transform(test)
    assert out["Age"].tolist() == [15.0, 55.0]


def test_preprocessor_leaves_no_nulls(toy):
    """Paso 46: tras el preprocesamiento no puede quedar ningún NaN."""
    X, y = toy
    pre = ml.build_pipeline(ml.get_candidates()["dummy"])[:-1].fit(X, y)
    Z = pre.transform(X)
    assert not np.isnan(Z).any()


def test_onehot_avoids_implicit_order(toy):
    """Paso 46: Sex sale como dos columnas 0/1, no como un numero con orden."""
    X, y = toy
    pre = ml.build_pipeline(ml.get_candidates()["dummy"])[:-1].fit(X, y)
    nombres = list(pre["prep"].get_feature_names_out())
    assert "cat__Sex_female" in nombres
    assert "cat__Sex_male" in nombres


def test_pipeline_learns_nothing_from_data_it_does_not_see(toy):
    """El nucleo del Paso 43: el pipeline ajustado en train no mira test.

    Se ajusta dos veces con el mismo train pero distintos test, y las
    predicciones sobre un tercer conjunto deben ser identicas.
    """
    X, y = toy
    X_tr, X_te = X.iloc[:80], X.iloc[80:]
    y_tr = y.iloc[:80]
    pipe_a = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X_tr, y_tr)
    pipe_b = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X_tr, y_tr)
    np.testing.assert_allclose(
        pipe_a.predict_proba(X_te)[:, 1], pipe_b.predict_proba(X_te)[:, 1]
    )


def test_rare_categories_are_grouped(toy):
    """min_frequency evita una columna casi vacia por cada categoria rara."""
    X, y = toy
    pre = ml.build_pipeline(ml.get_candidates()["dummy"])[:-1].fit(X, y)
    nombres = [n for n in pre["prep"].get_feature_names_out() if n.startswith("cat__TCabin")]
    assert len(nombres) < X["TCabin"].nunique() + 1


def test_every_candidate_fits_inside_the_pipeline(toy):
    """Los seis candidatos deben poder entrenarse con el mismo preprocesamiento."""
    X, y = toy
    for nombre, modelo in ml.get_candidates(42).items():
        pipe = ml.build_pipeline(modelo).fit(X, y)
        assert len(pipe.predict(X)) == len(X), nombre


# --- Parte 12: validación cruzada -------------------------------------------

def test_cross_validate_returns_one_row_per_candidate(toy):
    X, y = toy
    tabla = ml.cross_validate_candidates(X, y, cv_folds=3)
    assert set(tabla["modelo"]) == set(ml.get_candidates())
    assert len(tabla) == len(ml.get_candidates())


def test_cross_validate_reports_mean_and_std_of_every_metric(toy):
    """La desviacion importa tanto como la media: mide estabilidad."""
    X, y = toy
    tabla = ml.cross_validate_candidates(X, y, cv_folds=3)
    for metrica in ml.SCORING:
        assert f"{metrica}_mean" in tabla.columns
        assert f"{metrica}_std" in tabla.columns


def test_cross_validate_is_sorted_by_the_primary_metric(toy):
    X, y = toy
    col = f"{ml.PRIMARY_METRIC}_mean"
    valores = ml.cross_validate_candidates(X, y, cv_folds=3)[col].tolist()
    assert valores == sorted(valores, reverse=True)


def test_cross_validate_reports_the_overfitting_gap(toy):
    """Entrenamiento menos validacion: cuanto peor le va con datos que no vio."""
    X, y = toy
    tabla = ml.cross_validate_candidates(X, y, cv_folds=3).set_index("modelo")
    assert f"brecha_{ml.PRIMARY_METRIC}" in tabla.columns
    esperada = (tabla[f"train_{ml.PRIMARY_METRIC}_mean"]
                - tabla[f"{ml.PRIMARY_METRIC}_mean"])
    np.testing.assert_allclose(tabla[f"brecha_{ml.PRIMARY_METRIC}"], esperada, atol=1e-4)


def test_dummy_scores_half_in_roc_auc(toy):
    """El baseline no ordena nada: su ROC-AUC es el del azar."""
    X, y = toy
    tabla = ml.cross_validate_candidates(X, y, cv_folds=3).set_index("modelo")
    assert tabla.loc["dummy", "roc_auc_mean"] == pytest.approx(0.5, abs=0.01)
    assert tabla.loc["dummy", "recall_mean"] == 0.0


def test_cross_validate_can_return_per_fold_scores(toy):
    """El Paso 49 necesita el detalle por pliegue para el boxplot."""
    X, y = toy
    tabla, pliegues = ml.cross_validate_candidates(X, y, cv_folds=3, return_folds=True)
    assert list(pliegues.columns) == ["modelo", "metrica", "pliegue", "valor"]
    assert len(pliegues) == len(tabla) * len(ml.SCORING) * 3
    assert set(pliegues["pliegue"]) == {0, 1, 2}


def test_cross_validate_is_reproducible(toy):
    """Misma semilla, mismos resultados: sin eso no se pueden comparar modelos."""
    X, y = toy
    col = f"{ml.PRIMARY_METRIC}_mean"
    a = ml.cross_validate_candidates(X, y, cv_folds=3, random_state=7)[col].tolist()
    b = ml.cross_validate_candidates(X, y, cv_folds=3, random_state=7)[col].tolist()
    assert a == b


def test_cross_validate_accepts_a_custom_feature_set(toy):
    """El Paso 61 compara con y sin las variables sospechosas de cabina."""
    X, y = toy
    tabla = ml.cross_validate_candidates(
        X, y, cv_folds=3,
        cat_features=ml.CAT_FEATURES + ml.CABIN_CAT,
        num_features=ml.NUM_FEATURES + ml.CABIN_NUM,
    )
    assert len(tabla) == len(ml.get_candidates())


# --- Parte 13: evaluación en profundidad -------------------------------------

@pytest.fixture
def oof(toy):
    """Probabilidades fuera de muestra de un modelo lineal sobre el toy."""
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates()["logreg"])
    return X, y, ml.oof_proba(pipe, X, y, cv_folds=3)


def test_make_cv_defaults_come_from_settings():
    cv = ml.make_cv()
    assert cv.n_splits == ml.settings.cv_folds
    assert cv.random_state == ml.settings.random_state
    assert cv.shuffle is True


def test_make_cv_gives_identical_folds_across_calls(toy):
    """Si los pliegues cambiaran, el Paso 51 no sería comparable con el 48."""
    X, y = toy
    a = [te.tolist() for _, te in ml.make_cv(cv_folds=3).split(X, y)]
    b = [te.tolist() for _, te in ml.make_cv(cv_folds=3).split(X, y)]
    assert a == b


def test_oof_proba_gives_one_probability_per_passenger(oof):
    _X, y, proba = oof
    assert proba.shape == (len(y),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_oof_proba_is_not_fitted_on_the_passenger_it_predicts(toy):
    """Fuera de muestra: peor que en entrenamiento, y nunca perfecto."""
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates()["tree"])
    fuera = ml.oof_proba(pipe, X, y, cv_folds=3)
    dentro = pipe.fit(X, y).predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, fuera) < roc_auc_score(y, dentro)


def test_oof_proba_finalists_has_one_column_per_model(toy):
    X, y = toy
    tabla = ml.oof_proba_finalists(X, y, names=["logreg", "tree"], cv_folds=3)
    assert list(tabla.columns) == ["logreg", "tree"]
    assert len(tabla) == len(y)


def test_confusion_table_cells_add_up_to_everyone(oof):
    _, y, proba = oof
    tabla = ml.confusion_table(y, (proba >= 0.5).astype(int))
    assert tabla.to_numpy().sum() == len(y)
    assert tabla.shape == (2, 2)
    assert "real: sobrevivió" in tabla.index


def test_confusion_table_keeps_both_classes_with_a_degenerate_prediction(oof):
    """Si el modelo no predice positivos, la matriz sigue siendo 2x2."""
    _, y, _ = oof
    tabla = ml.confusion_table(y, np.zeros(len(y), dtype=int))
    assert tabla.shape == (2, 2)
    assert tabla["predicho: sobrevivió"].sum() == 0


def test_threshold_sweep_columns_have_the_same_length(oof):
    """precision_recall_curve devuelve un punto más que umbrales."""
    _, y, proba = oof
    barrido = ml.threshold_sweep(y, proba)
    assert list(barrido.columns) == ["umbral", "precision", "recall", "f1"]
    assert barrido.notna().all().all()
    assert barrido["f1"].between(0, 1).all()


def test_threshold_sweep_recall_falls_as_the_threshold_rises(oof):
    _, y, proba = oof
    barrido = ml.threshold_sweep(y, proba).sort_values("umbral")
    assert barrido["recall"].is_monotonic_decreasing


def test_best_threshold_beats_half_on_its_own_criterion(oof):
    """El umbral elegido no puede tener peor F1 que el 0.5 por convención."""
    _, y, proba = oof
    from sklearn.metrics import f1_score
    mejor = ml.best_threshold(y, proba, criterio="f1")
    assert mejor["f1"] >= round(f1_score(y, (proba >= 0.5).astype(int)), 4) - 1e-9
    assert 0 <= mejor["umbral"] <= 1


def test_threshold_for_precision_reaches_the_floor(oof):
    _, y, proba = oof
    r = ml.threshold_for_precision(y, proba, min_precision=0.70)
    assert r["alcanzado"] is True
    assert r["precision"] >= 0.70


def test_threshold_for_precision_flags_an_unreachable_floor(oof):
    """Con un piso imposible devuelve el mejor disponible, marcado como fallido."""
    _, y, proba = oof
    r = ml.threshold_for_precision(y, proba, min_precision=1.01)
    assert r["alcanzado"] is False


def test_calibration_table_counts_every_passenger_once(oof):
    _, y, proba = oof
    tabla = ml.calibration_table(y, proba, n_bins=4)
    assert len(tabla) == 4
    assert tabla["n"].sum() == len(y)
    assert {"tramo", "n", "proba_media", "tasa_real", "desvio"} == set(tabla.columns)


def test_calibration_table_keeps_empty_bins(oof):
    """calibration_curve los descarta; sin ellos no se ve dónde no hay datos."""
    _, y, proba = oof
    assert len(ml.calibration_table(y, proba, n_bins=20)) == 20


def test_brier_is_zero_for_a_perfect_forecast():
    y = np.array([0, 1, 0, 1])
    assert ml.brier(y, y.astype(float)) == 0.0


def test_brier_matches_the_base_rate_forecast():
    """Predecir siempre la tasa base da p*(1-p): la vara real, no 0.25."""
    y = np.r_[np.zeros(60), np.ones(40)]
    assert ml.brier(y, np.full(100, 0.4)) == round(0.4 * 0.6, 4)


def test_subgroup_report_splits_the_whole_sample(oof):
    X, y, proba = oof
    tabla = ml.subgroup_report(X, y, (proba >= 0.5).astype(int), by="Sex")
    assert tabla["n"].sum() == len(y)
    assert set(tabla["grupo"]) == set(X["Sex"])
    assert (tabla["variable"] == "Sex").all()


def test_subgroup_report_is_sorted_worst_first(oof):
    """El subgrupo peor servido va arriba: es lo que hay que mirar."""
    X, y, proba = oof
    tabla = ml.subgroup_report(X, y, (proba >= 0.5).astype(int), by="Pclass")
    assert tabla["exactitud"].is_monotonic_increasing


def test_confident_errors_returns_only_errors(oof):
    X, y, proba = oof
    errores = ml.confident_errors(X, y, proba, n=5)
    assert len(errores) <= 5
    assert (errores["pred"] != errores["real"]).all()
    assert errores["confianza"].is_monotonic_decreasing


def test_confident_errors_does_not_mutate_the_features(oof):
    X, y, proba = oof
    antes = list(X.columns)
    ml.confident_errors(X, y, proba)
    assert list(X.columns) == antes


def test_learning_curve_table_grows_with_the_sample(toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates()["logreg"])
    tabla = ml.learning_curve_table(pipe, X, y, cv_folds=3)
    assert len(tabla) == 5
    assert tabla["n_entrenamiento"].is_monotonic_increasing
    assert (tabla["brecha"] == (tabla["train_mean"] - tabla["val_mean"]).round(4)).all()


def test_finalists_are_real_candidates():
    """Los nombres del Paso 50 tienen que existir en get_candidates()."""
    assert set(ml.FINALISTS) <= set(ml.get_candidates())
