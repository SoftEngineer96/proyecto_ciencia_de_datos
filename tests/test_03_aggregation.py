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
    """Silver mínima: 8 pasajeros con una diferencia clara por sexo."""
    return pd.DataFrame({
        "PassengerId": range(1, 9),
        "Survived": [1, 1, 1, 0, 0, 0, 0, 1],
        "Pclass": [1, 1, 3, 3, 3, 2, 2, 1],
        "Sex": ["female", "female", "female", "male", "male", "male", "male", "female"],
        "Age": [38.0, 26.0, 4.0, 22.0, 35.0, 54.0, 2.0, 58.0],
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


def test_pending_part9_functions_fail_loudly(silver):
    """Mejor un NotImplementedError que una tabla vacía que parezca válida."""
    for fn in (ag.survival_by_class_sex, ag.survival_by_embarked,
               ag.survival_by_family, ag.outlier_summary):
        with pytest.raises(NotImplementedError):
            fn(silver)


def test_outlier_mask_iqr_is_still_pending():
    with pytest.raises(NotImplementedError):
        ag.outlier_mask_iqr(pd.Series(np.arange(10.0)))
