"""Pruebas de la capa Gold.

Se añaden a medida que se implementan las funciones de flows/03_aggregation.py.
Las de la Parte 9 (Pasos 34 a 38) llegan con sus pasos.
"""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

ag = importlib.import_module("flows.03_aggregation")


@pytest.fixture
def silver():
    """Silver mínima: 8 pasajeros con una diferencia clara por sexo.

    Trae todas las columnas que run() necesita, porque desde la Parte 9 la
    etapa produce también las tablas de puerto, tarifa, familia y atípicos.
    """
    return pd.DataFrame({
        "PassengerId": range(1, 9),
        "Survived": [1, 1, 1, 0, 0, 0, 0, 1],
        "Pclass": [1, 1, 3, 3, 3, 2, 2, 1],
        "Sex": ["female", "female", "female", "male", "male", "male", "male", "female"],
        "Age": [38.0, 26.0, 4.0, 22.0, 35.0, 54.0, 2.0, 58.0],
        "AgeOriginal": [38.0, 26.0, 4.0, 22.0, 35.0, 54.0, 2.0, 58.0],
        "Embarked": ["S", "C", "S", "Q", "S", "S", "C", "Q"],
        "Fare": [80.0, 90.0, 7.0, 7.5, 8.0, 13.0, 14.0, 300.0],
        "Familia": [0, 1, 1, 0, 0, 0, 1, 0],
        "Alone": ["Alone", "with Family", "with Family", "Alone",
                  "Alone", "Alone", "with Family", "Alone"],
    })


# --- Paso 28 ---------------------------------------------------------------

def test_group_stats_returns_the_four_tables(silver):
    tablas = ag.group_stats(silver)
    assert set(tablas) == {
        "age_by_survival", "survival_by_sex", "survival_by_pclass", "survival_by_alone",
    }
    assert all(isinstance(t, pd.DataFrame) for t in tablas.values())


def test_group_stats_tables_have_a_flat_index(silver):
    """Parquet guarda el índice aparte: todas las tablas salen con reset_index()."""
    for nombre, tabla in ag.group_stats(silver).items():
        assert isinstance(tabla.index, pd.RangeIndex), nombre
        assert all(isinstance(c, str) for c in tabla.columns), nombre


def test_survival_by_sex_reports_rate_and_size(silver):
    """La tasa sola engaña: 'count' tiene que venir con ella."""
    tabla = ag.group_stats(silver)["survival_by_sex"].set_index("Sex")
    assert list(tabla.columns) == ["mean", "sum", "count"]
    assert tabla.loc["female", "mean"] == pytest.approx(1.0)
    assert tabla.loc["female", "count"] == 4
    assert tabla.loc["male", "mean"] == pytest.approx(0.0)
    assert tabla.loc["male", "count"] == 4


def test_age_by_survival_describes_both_groups(silver):
    tabla = ag.group_stats(silver)["age_by_survival"]
    assert len(tabla) == 2
    assert "mean" in tabla.columns and "count" in tabla.columns


# --- Paso 29 ---------------------------------------------------------------

def test_chi2_returns_the_three_keys(silver):
    r = ag.chi2_sex_survival(silver)
    assert set(r) == {"chi2", "p_value", "dof"}
    assert r["dof"] == 1                      # tabla 2x2 -> (2-1)(2-1)
    assert 0.0 <= r["p_value"] <= 1.0


def test_chi2_is_json_serializable(silver):
    """scipy devuelve tipos de NumPy; json.dumps no los serializa."""
    json.dumps(ag.chi2_sex_survival(silver))


def test_chi2_detects_a_strong_association(silver):
    """Con separación perfecta por sexo, el p-valor debe ser pequeño."""
    assert ag.chi2_sex_survival(silver)["p_value"] < 0.05


def test_chi2_does_not_flag_independence_as_association():
    """Si el sexo no informa nada, el p-valor debe ser alto."""
    sin_relacion = pd.DataFrame({
        "Sex": ["female", "male"] * 50,
        "Survived": [1, 1, 0, 0] * 25,
    })
    assert ag.chi2_sex_survival(sin_relacion)["p_value"] > 0.05


# --- Orquestación ----------------------------------------------------------

def test_run_writes_the_tables_and_the_json(tmp_path, silver):
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    silver.to_parquet(silver_dir / "titanic_silver.parquet", index=False)

    out = tmp_path / "gold"
    ag.run(in_dir=silver_dir, out_dir=out)

    for nombre in ("age_by_survival", "survival_by_sex",
                   "survival_by_pclass", "survival_by_alone"):
        assert (out / f"{nombre}.parquet").exists(), nombre

    prueba = json.loads((out / "chi2_sex_survival.json").read_text(encoding="utf-8"))
    assert set(prueba) == {"chi2", "p_value", "dof"}


def test_gold_tables_survive_a_parquet_round_trip(tmp_path, silver):
    """El gotcha de Parquet: columnas de texto e índice plano."""
    out = tmp_path / "gold"
    out.mkdir()
    for nombre, tabla in ag.group_stats(silver).items():
        ruta = out / f"{nombre}.parquet"
        tabla.to_parquet(ruta, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(ruta), tabla)


# --- Parte 9: pruebas del enunciado ----------------------------------------

@pytest.fixture
def silver_like():
    """Fixture del enunciado (Paso 38)."""
    return pd.DataFrame({
        "Survived": [1, 1, 0, 0, 1, 0, 0, 1],
        "Pclass": [1, 1, 1, 3, 3, 3, 3, 3],
        "Sex": ["female", "male", "male", "female", "female", "male", "male", "male"],
        "Embarked": ["S", "C", "S", "Q", "S", "S", "C", "Q"],
        "Fare": [80.0, 90.0, 70.0, 7.0, 8.0, 7.5, 9.0, 300.0],
        "Age": [30.0, 40.0, 50.0, 20.0, 25.0, 30.0, 35.0, 28.0],
        "Familia": [0, 1, 0, 2, 0, 0, 1, 0],
        "Alone": ["Alone", "with Family", "Alone", "with Family", "Alone", "Alone",
                  "with Family", "Alone"],
    })


def test_outlier_mask_iqr_flags_only_extreme_values():
    """Prueba del enunciado."""
    mask = ag.outlier_mask_iqr(pd.Series([1, 2, 3, 4, 5, 100]))
    assert mask.tolist() == [False, False, False, False, False, True]


def test_survival_by_class_sex_is_a_valid_rate_table(silver_like):
    """Prueba del enunciado."""
    out = ag.survival_by_class_sex(silver_like)
    assert "Pclass" in out.columns
    rates = out.drop(columns="Pclass")
    assert ((rates.dropna() >= 0) & (rates.dropna() <= 1)).all().all()
    assert all(isinstance(c, str) for c in out.columns)      # requisito de Parquet


def test_fare_quartiles_are_computed_within_class(silver_like):
    """Prueba del enunciado."""
    out = ag.add_fare_quartile(silver_like)
    for _, group in out.groupby("Pclass"):
        assert group["FareQ"].between(0, 3).all()
    assert "FareQ" not in silver_like.columns                 # no muta la entrada


def test_chi2_returns_valid_p_value(silver_like):
    """Prueba del enunciado."""
    res = ag.chi2_sex_survival(silver_like)
    assert set(res) == {"chi2", "p_value", "dof"}
    assert 0 <= res["p_value"] <= 1


def test_run_writes_gold_tables(tmp_path, silver_like):
    """Prueba del enunciado."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    silver_like.to_parquet(silver_dir / "titanic_silver.parquet", index=False)
    out = ag.run(in_dir=silver_dir, out_dir=tmp_path / "gold")
    assert (out / "survival_by_class_sex.parquet").exists()
    assert (out / "chi2_sex_survival.json").exists()


# --- Parte 9: pruebas complementarias --------------------------------------

def test_outlier_mask_ignores_nulls():
    """Una comparación con NaN da False: los nulos no se marcan como atípicos."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 100.0])
    assert ag.outlier_mask_iqr(s).tolist() == [False] * 5 + [False, True]


def test_stratifying_separates_the_embarked_effect(silver_like):
    """El puerto sin estratificar y estratificado son tablas distintas."""
    plano = ag.survival_by_embarked(silver_like)
    estratificado = ag.survival_by_embarked_class(silver_like)
    assert list(plano.columns) == ["Embarked", "mean", "count"]
    assert "Embarked" in estratificado.columns
    assert all(isinstance(c, str) for c in estratificado.columns)


def test_survival_by_family_reports_group_size(silver_like):
    """Sin 'count' no se sabe si una tasa del 100 % son 50 pasajeros o 1."""
    out = ag.survival_by_family(silver_like)
    assert list(out.columns) == ["Familia", "mean", "count"]
    assert out["count"].sum() == len(silver_like)


def test_outlier_summary_prefers_age_original():
    """La deteccion sobre la edad imputada marcaria pasajeros normales (H17)."""
    base = pd.DataFrame({
        "Survived": [1, 0, 1, 0, 1, 0],
        "Fare": [10.0, 12.0, 11.0, 13.0, 9.0, 500.0],
        "Familia": [0, 1, 0, 1, 0, 0],
        "Age": [30.0] * 6,
        "AgeOriginal": [30.0, 31.0, 29.0, 30.0, 28.0, 99.0],
    })
    resumen = ag.outlier_summary(base)
    fila_edad = resumen.set_index("columna").loc["Age"]
    assert fila_edad["calculada_sobre"] == "AgeOriginal"
    assert fila_edad["n_atipicos"] == 1


def test_outlier_summary_falls_back_to_age(silver_like):
    """Sin AgeOriginal usa Age, para que run() funcione sobre cualquier Silver."""
    resumen = ag.outlier_summary(silver_like)
    assert resumen.set_index("columna").loc["Age", "calculada_sobre"] == "Age"


def test_outlier_summary_compares_survival_rates(silver_like):
    """El enunciado pide la tasa de los atípicos de Fare frente al resto."""
    resumen = ag.outlier_summary(silver_like).set_index("columna")
    assert {"tasa_atipicos", "tasa_resto"} <= set(resumen.columns)
    assert resumen.loc["Fare", "n_atipicos"] == 1          # la tarifa de 300


def test_all_gold_tables_survive_parquet(tmp_path, silver_like):
    """Ninguna tabla puede llevar nombres de columna no textuales."""
    out = tmp_path / "gold"
    tablas = {
        "survival_by_class_sex": ag.survival_by_class_sex(silver_like),
        "survival_by_embarked": ag.survival_by_embarked(silver_like),
        "survival_by_embarked_class": ag.survival_by_embarked_class(silver_like),
        "survival_by_fare_quartile_class": ag.survival_by_fare_quartile_class(silver_like),
        "survival_by_family": ag.survival_by_family(silver_like),
        "outlier_summary": ag.outlier_summary(silver_like),
    }
    out.mkdir()
    for nombre, tabla in tablas.items():
        assert all(isinstance(c, str) for c in tabla.columns), nombre
        ruta = out / f"{nombre}.parquet"
        tabla.to_parquet(ruta, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(ruta), tabla)
