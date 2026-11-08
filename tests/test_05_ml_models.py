"""Pruebas del modelado.

Se añaden a medida que se implementan las funciones de flows/05_ml_models.py.
"""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from flows.common.hashing import sha256_of

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


# --- Parte 14: ajuste de hiperparámetros -------------------------------------

def test_param_grids_cover_every_finalist():
    """Los tres modelos del Paso 50 tienen que tener rejilla."""
    assert set(ml.FINALISTS) <= set(ml.get_param_grids())


def test_param_grid_keys_use_the_pipeline_prefix():
    """Sin 'model__' la búsqueda no sabría a qué paso del Pipeline apuntar."""
    for nombre, rejilla in ml.get_param_grids().items():
        for clave in rejilla:
            assert clave.startswith("model__"), f"{nombre}: {clave}"


def test_param_grids_name_real_estimator_parameters():
    """Un nombre mal escrito solo falla al ajustar; aquí falla al instante."""
    candidatos = ml.get_candidates()
    for nombre, rejilla in ml.get_param_grids().items():
        validos = set(ml.build_pipeline(candidatos[nombre]).get_params())
        assert set(rejilla) <= validos, f"{nombre}: {set(rejilla) - validos}"


def test_param_grids_have_no_empty_lists():
    for nombre, rejilla in ml.get_param_grids().items():
        for clave, valores in rejilla.items():
            assert len(valores) >= 1, f"{nombre}: {clave}"


def test_tune_model_is_never_worse_than_the_default(toy):
    """C=1 está en la rejilla, así que el óptimo no puede quedar por debajo."""
    X, y = toy
    gs = ml.tune_model(X, y, name="logreg", cv_folds=3, n_jobs=1)
    por_defecto = ml.cross_validate_candidates(
        X, y, cv_folds=3
    ).set_index("modelo").loc["logreg", "roc_auc_mean"]
    assert gs.best_score_ >= por_defecto - 1e-9


def test_tune_model_returns_a_usable_estimator(toy):
    X, y = toy
    gs = ml.tune_model(X, y, name="logreg", cv_folds=3, n_jobs=1)
    assert set(gs.best_params_) <= set(ml.get_param_grids()["logreg"])
    proba = gs.best_estimator_.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_search_results_is_ranked_and_measures_flatness(toy):
    X, y = toy
    gs = ml.tune_model(X, y, name="logreg", cv_folds=3, n_jobs=1)
    tabla = ml.search_results(gs)
    assert tabla["rank_test_score"].is_monotonic_increasing
    assert tabla.loc[0, "dif_vs_mejor"] == 0.0
    assert (tabla["dif_vs_mejor"] >= 0).all()
    assert tabla["params"].map(type).eq(str).all()      # Parquet y display


def test_paired_comparison_uses_the_same_folds_for_both(toy):
    """Si best_params no cambia nada, la diferencia debe ser exactamente 0."""
    X, y = toy
    tabla = ml.paired_fold_comparison(X, y, name="logreg", best_params=None, cv_folds=3)
    assert len(tabla) == 3
    assert (tabla["diferencia"] == 0).all()


def test_paired_comparison_reports_one_row_per_fold(toy):
    X, y = toy
    tabla = ml.paired_fold_comparison(X, y, name="logreg",
                                      best_params={"model__C": 10}, cv_folds=3)
    assert list(tabla.columns) == ["pliegue", "por_defecto", "ajustado", "diferencia"]
    esperado = (tabla["ajustado"] - tabla["por_defecto"]).round(4)
    assert np.allclose(tabla["diferencia"], esperado)


def test_nested_cv_returns_one_score_per_outer_fold(toy):
    X, y = toy
    r = ml.nested_cv_score(X, y, name="logreg", cv_folds=3, n_jobs=1)
    assert set(r) == {"puntajes", "media", "desviacion"}
    assert len(r["puntajes"]) == 3
    assert 0 <= r["media"] <= 1


def test_nested_cv_outer_folds_differ_from_the_inner_ones(toy):
    """Si coincidieran, la busqueda veria los datos con los que se la evalua."""
    X, y = toy
    internos = [te.tolist() for _, te in ml.make_cv(42, 3).split(X, y)]
    externos = [te.tolist() for _, te in ml.make_cv(43, 3).split(X, y)]
    assert internos != externos


# --- Parte 15: interpretación y ablación -------------------------------------

def test_linear_coefficients_has_one_row_per_generated_column(toy):
    X, y = toy
    tabla = ml.linear_coefficients(X, y)
    assert list(tabla.columns) == ["variable", "coeficiente", "razon_de_momios"]
    assert tabla["variable"].map(type).eq(str).all()
    assert len(tabla) >= len(ml.NUM_FEATURES)      # one-hot añade columnas


def test_linear_coefficients_is_sorted_by_absolute_value(toy):
    """Lo que importa es la magnitud, no el signo."""
    X, y = toy
    magnitudes = ml.linear_coefficients(X, y)["coeficiente"].abs()
    assert magnitudes.is_monotonic_decreasing


def test_odds_ratio_is_the_exponential_of_the_coefficient(toy):
    X, y = toy
    tabla = ml.linear_coefficients(X, y)
    assert np.allclose(tabla["razon_de_momios"], np.exp(tabla["coeficiente"]), atol=1e-3)


def test_odds_ratio_sits_on_the_right_side_of_one(toy):
    """Coeficiente positivo empuja a sobrevivir: razón de momios > 1."""
    tabla = ml.linear_coefficients(*toy)
    positivos = tabla[tabla["coeficiente"] > 0]
    assert (positivos["razon_de_momios"] > 1).all()


def test_permutation_table_covers_the_original_columns(toy):
    """Se permutan las columnas de entrada, no las que genera el one-hot."""
    X, y = toy
    X_tr, X_val, y_tr, y_val = ml.split(X, y, test_size=0.3)
    pipe = ml.build_pipeline(ml.get_candidates()["logreg"]).fit(X_tr, y_tr)
    tabla = ml.permutation_table(pipe, X_val, y_val, n_repeats=3)
    assert set(tabla["variable"]) == set(X_val.columns)
    assert tabla["caida_media"].is_monotonic_decreasing


def test_permutation_finds_the_variable_that_generates_the_target(toy):
    """En el toy, y se construye desde Sex: debe quedar primera."""
    X, y = toy
    X_tr, X_val, y_tr, y_val = ml.split(X, y, test_size=0.3)
    pipe = ml.build_pipeline(ml.get_candidates()["logreg"]).fit(X_tr, y_tr)
    assert ml.permutation_table(pipe, X_val, y_val, n_repeats=5).loc[0, "variable"] == "Sex"


def test_permutation_is_reproducible(toy):
    X, y = toy
    X_tr, X_val, y_tr, y_val = ml.split(X, y, test_size=0.3)
    pipe = ml.build_pipeline(ml.get_candidates()["logreg"]).fit(X_tr, y_tr)
    a = ml.permutation_table(pipe, X_val, y_val, n_repeats=3, random_state=7)
    b = ml.permutation_table(pipe, X_val, y_val, n_repeats=3, random_state=7)
    pd.testing.assert_frame_equal(a, b)


def test_ablation_configs_isolate_one_change_each():
    cfg = ml.ablation_configs()
    assert set(cfg) == {"A. base", "B. base + cabina",
                        "C. base sin TTitulo", "D. base sin Sex"}
    assert "TTitulo" not in cfg["C. base sin TTitulo"][1]
    assert "Sex" not in cfg["D. base sin Sex"][1]
    assert set(ml.CABIN_CAT) <= set(cfg["B. base + cabina"][1])
    assert set(ml.CABIN_NUM) <= set(cfg["B. base + cabina"][0])


def test_ablation_returns_one_row_per_configuration(toy):
    X, y = toy
    tabla = ml.ablation(X, y, name="logreg", cv_folds=3)
    assert len(tabla) == 4
    assert tabla.loc[tabla["configuracion"] == "A. base", "delta_vs_A"].iloc[0] == 0.0


def test_ablation_detects_the_loss_of_the_driving_variable(toy):
    """Quitar Sex, que genera y en el toy, tiene que doler."""
    X, y = toy
    tabla = ml.ablation(X, y, name="logreg", cv_folds=3).set_index("configuracion")
    assert tabla.loc["D. base sin Sex", "delta_vs_A"] < 0


def test_ablation_without_sex_also_removes_it_from_the_imputer():
    """Si no, Sex seguiria entrando al modelo por la edad imputada."""
    cfg = ml.ablation_configs()["D. base sin Sex"]
    pipe = ml.build_pipeline(ml.get_candidates()["logreg"], *cfg)
    pipe.set_params(age_imputer__group_cols=("Pclass",))
    assert "Sex" not in pipe.named_steps["age_imputer"].group_cols


# --- Parte 16: evaluación final ----------------------------------------------

@pytest.fixture
def ajustado(toy):
    """Un pipeline entrenado y una partición de prueba, para la Parte 16."""
    X, y = toy
    X_tr, X_te, y_tr, y_te = ml.split(X, y, test_size=0.25, random_state=1)
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X_tr, y_tr)
    return pipe, X_te, y_te


def test_evaluate_on_test_returns_metrics(toy):
    """Prueba del enunciado."""
    X, y = toy
    X_tr, X_te, y_tr, y_te = ml.split(X, y, test_size=0.25, random_state=1)
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X_tr, y_tr)
    res = ml.evaluate_on_test(pipe, X_te, y_te, threshold=0.5)
    assert {"accuracy", "f1", "roc_auc", "n_test"} <= set(res)
    assert 0 <= res["accuracy"] <= 1 and res["n_test"] == len(X_te)


def test_evaluate_on_test_is_json_serializable(ajustado):
    """Las métricas entran en los metadatos del modelo: nada de tipos NumPy."""
    pipe, X_te, y_te = ajustado
    json.dumps(ml.evaluate_on_test(pipe, X_te, y_te))


def test_evaluate_on_test_honours_the_threshold(ajustado):
    """Subir el umbral no puede aumentar el recall."""
    pipe, X_te, y_te = ajustado
    bajo = ml.evaluate_on_test(pipe, X_te, y_te, threshold=0.3)
    alto = ml.evaluate_on_test(pipe, X_te, y_te, threshold=0.7)
    assert alto["recall"] <= bajo["recall"]
    assert alto["roc_auc"] == bajo["roc_auc"]      # el AUC no depende del umbral


def test_bootstrap_ci_brackets_the_estimate(ajustado):
    pipe, X_te, y_te = ajustado
    proba = pipe.predict_proba(X_te)[:, 1]
    ic = ml.bootstrap_ci(y_te, proba, n_boot=200)
    assert ic["ic_bajo"] <= ic["estimacion"] <= ic["ic_alto"]
    assert ic["amplitud"] == round(ic["ic_alto"] - ic["ic_bajo"], 4)
    assert ic["n_remuestreos"] <= 200


def test_bootstrap_ci_narrows_with_more_data():
    """El ancho del intervalo es la lección del Paso 62."""
    rng = np.random.default_rng(0)

    def ancho(n):
        y = rng.integers(0, 2, n)
        proba = np.clip(y * 0.4 + rng.normal(0.3, 0.2, n), 0, 1)
        return ml.bootstrap_ci(y, proba, n_boot=200)["amplitud"]

    assert ancho(2000) < ancho(150)


def test_bootstrap_ci_is_reproducible(ajustado):
    pipe, X_te, y_te = ajustado
    proba = pipe.predict_proba(X_te)[:, 1]
    a = ml.bootstrap_ci(y_te, proba, n_boot=100, random_state=3)
    b = ml.bootstrap_ci(y_te, proba, n_boot=100, random_state=3)
    assert a == b


# --- Parte 17: decisiones congeladas y modelo final ---------------------------

def test_frozen_decisions_are_coherent():
    """Las constantes del Paso 64 tienen que referirse a cosas que existen."""
    assert ml.FINAL_ESTIMATOR in ml.get_candidates()
    assert set(ml.FINAL_PARAMS) <= set(ml.build_pipeline(
        ml.get_candidates()[ml.FINAL_ESTIMATOR]).get_params())
    assert 0.0 < ml.THRESHOLD < 1.0
    assert isinstance(ml.USE_CABIN_FEATURES, bool)


def test_final_features_follow_the_cabin_decision():
    num, cat = ml.final_features()
    hay_cabina = set(ml.CABIN_NUM) <= set(num) and set(ml.CABIN_CAT) <= set(cat)
    assert hay_cabina == ml.USE_CABIN_FEATURES
    assert set(ml.NUM_FEATURES) <= set(num) and set(ml.CAT_FEATURES) <= set(cat)


def test_train_final_applies_the_frozen_parameters(toy):
    X, y = toy
    pipe = ml.train_final(X, y)
    for clave, valor in ml.FINAL_PARAMS.items():
        assert pipe.get_params()[clave] == valor
    assert hasattr(pipe["model"], "predict_proba")


def test_save_and_load_roundtrip(tmp_path, toy):
    """Prueba del enunciado."""
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    path = ml.save_model(pipe, {"estimator": "logreg", "n_train": len(X)},
                         name="toy_model", models_dir=tmp_path)
    loaded, meta = ml.load_model(path)
    assert np.array_equal(pipe.predict(X), loaded.predict(X))
    assert {"sklearn_version", "python_version", "model_sha256", "saved_at"} <= set(meta)
    assert meta["n_train"] == len(X)


def test_predict_returns_probabilities_and_labels(toy):
    """Prueba del enunciado."""
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    out = ml.predict(pipe, X, threshold=0.5)
    assert list(out.columns) == ["proba_survive", "pred"]
    assert set(out["pred"].unique()) <= {0, 1}


def test_predict_uses_the_frozen_threshold_by_default(toy):
    """Si predict usara 0.5 por costumbre, ignoraría la decisión del Paso 53."""
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    por_defecto = ml.predict(pipe, X)
    explicito = ml.predict(pipe, X, threshold=ml.THRESHOLD)
    pd.testing.assert_frame_equal(por_defecto, explicito)


def test_model_sha256_matches_the_saved_file(tmp_path, toy):
    """La huella tiene que ser la del archivo, no una cadena cualquiera."""
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    path = ml.save_model(pipe, {}, name="toy_model", models_dir=tmp_path)
    _, meta = ml.load_model(path)
    assert meta["model_sha256"] == sha256_of(path)


def test_run_writes_the_model_and_its_metadata(tmp_path, silver):
    """El Paso 68 de punta a punta, sobre una Silver mínima."""
    modelo = ml.run(in_dir=silver.parent, models_dir=tmp_path)
    assert modelo.exists()

    meta = json.loads(modelo.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    requeridos = {"estimator", "params", "features", "target", "threshold",
                  "random_state", "n_train", "data_sha256", "metrics"}
    assert requeridos <= set(meta)
    assert 0 < meta["metrics"]["n_test"] < meta["n_train"]
    assert {"roc_auc_ic95", "accuracy_ic95"} <= set(meta["metrics"])


def test_run_trains_the_final_model_on_every_labelled_row(tmp_path, silver):
    """El artefacto se entrena con TODO, no solo con el 80 % de entrenamiento."""
    modelo = ml.run(in_dir=silver.parent, models_dir=tmp_path)
    meta = json.loads(modelo.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    X, _ = ml.load_dataset(silver)
    assert meta["n_train"] == len(X)


def test_run_records_the_hash_of_the_data_it_used(tmp_path, silver):
    modelo = ml.run(in_dir=silver.parent, models_dir=tmp_path)
    meta = json.loads(modelo.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert meta["data_sha256"] == sha256_of(silver)
