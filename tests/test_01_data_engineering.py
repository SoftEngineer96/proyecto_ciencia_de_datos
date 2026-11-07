"""Pruebas de la capa Silver.

Se añaden a medida que se implementan las funciones de flows/01_data_engineering.py.
Las pruebas del enunciado para add_person, add_cabin_features, add_family_features
y los títulos llegan con sus pasos correspondientes (8, 11, 14, 24, 25, 26).
"""
import importlib

import numpy as np
import pandas as pd
import pytest

de = importlib.import_module("flows.01_data_engineering")


@pytest.fixture
def sample():
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5, 6],
        "Survived": [0, 1, 1, 0, 1, 0],
        "Pclass": [3, 1, 3, 1, 2, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Palsson, Master. Gosta Leonard",
            "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)",
            "Moran, Dr. James",
        ],
        "Sex": ["male", "female", "female", "male", "female", "male"],
        "Age": [22.0, 38.0, np.nan, 2.0, 33.0, np.nan],
        "SibSp": [1, 1, 0, 3, 0, 0],
        "Parch": [0, 0, 0, 1, 0, 0],
        "Ticket": list("ABCDEF"),
        "Fare": [7.25, 71.28, 7.92, 21.07, 86.5, 8.46],
        "Cabin": [None, "C85", None, None, "B77", None],
        "Embarked": ["S", "C", "S", None, "S", "Q"],
    })


# --- Contrato de entrada (Paso 33) ----------------------------------------

def test_validate_bronze_rejects_missing_columns(sample):
    with pytest.raises(ValueError):
        de.validate_bronze(sample.drop(columns=["Fare"]))


def test_validate_bronze_rejects_duplicate_ids(sample):
    duplicado = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        de.validate_bronze(duplicado)


def test_validate_bronze_accepts_valid_input(sample):
    # Debe devolver el mismo objeto para poder encadenarla con .pipe().
    assert de.validate_bronze(sample) is sample


# --- Tipos (Paso 33) -------------------------------------------------------

def test_cast_types_fixes_numeric_columns(sample):
    entrada = sample.assign(Survived=sample["Survived"].astype(float))
    out = de.cast_types(entrada)
    assert out["Survived"].dtype.kind == "i"
    assert out["Pclass"].dtype.kind == "i"
    assert out["Age"].dtype.kind == "f"
    assert out["Fare"].dtype.kind == "f"


def test_cast_types_does_not_mutate_input(sample):
    """Las funciones de esta capa son puras: no alteran lo que reciben."""
    entrada = sample.assign(Survived=sample["Survived"].astype(float))
    antes = entrada["Survived"].dtype
    de.cast_types(entrada)
    assert entrada["Survived"].dtype == antes


def test_cast_types_keeps_age_nulls(sample):
    """Age tiene faltantes: convertirla a int los destruiría."""
    out = de.cast_types(sample)
    assert out["Age"].isna().sum() == 2


# --- Person (Paso 8) -------------------------------------------------------

def test_add_person_is_explicit_about_unknown_age(sample):
    """Prueba del enunciado: los dos pasajeros sin edad conservan su sexo."""
    out = de.add_person(sample)
    assert out["Person"].tolist() == ["male", "female", "female", "child", "female", "male"]


def test_add_person_uses_original_age_not_imputed(sample):
    """NaN < 16 es False: sin edad nunca hay 'child'. Regla documentada, no accidental."""
    out = de.add_person(sample)
    sin_edad = out[out["Age"].isna()]
    assert (sin_edad["Person"] == sin_edad["Sex"]).all()
    assert "child" not in sin_edad["Person"].tolist()


def test_add_person_does_not_mutate_input(sample):
    de.add_person(sample)
    assert "Person" not in sample.columns


# --- Cabina (Paso 11) ------------------------------------------------------

def test_cabin_features(sample):
    """Prueba del enunciado."""
    out = de.add_cabin_features(sample)
    assert out["TCabin"].tolist() == ["X", "C", "X", "X", "B", "X"]
    assert out["HasCabin"].tolist() == [0, 1, 0, 0, 1, 0]


def test_cabin_features_keep_the_original_column(sample):
    """Silver conserva Cabin: nunca se descarta información en silencio."""
    out = de.add_cabin_features(sample)
    assert "Cabin" in out.columns
    assert out["Cabin"].isna().sum() == 4


def test_has_cabin_matches_tcabin(sample):
    """HasCabin == 0 si y solo si TCabin == 'X'. Las dos columnas son coherentes."""
    out = de.add_cabin_features(sample)
    assert ((out["TCabin"] == "X") == (out["HasCabin"] == 0)).all()


# --- Familia (Paso 14) -----------------------------------------------------

def test_family_features(sample):
    """Prueba del enunciado."""
    out = de.add_family_features(sample)
    assert out["Familia"].tolist() == [1, 1, 0, 4, 0, 0]
    assert out["Alone"].tolist() == ["with Family", "with Family", "Alone",
                                     "with Family", "Alone", "Alone"]


def test_alone_is_consistent_with_familia(sample):
    out = de.add_family_features(sample)
    assert ((out["Familia"] == 0) == (out["Alone"] == "Alone")).all()


# --- Pureza de todas las transformaciones ya implementadas -----------------

def test_functions_do_not_mutate_their_input(sample):
    """Prueba del enunciado, adaptada a las funciones ya implementadas."""
    before = sample.copy(deep=True)
    for fn in (de.add_person, de.add_family_features, de.add_cabin_features, de.cast_types):
        fn(sample)
    pd.testing.assert_frame_equal(sample, before)


# --- Salida ----------------------------------------------------------------

def test_drop_original_columns_returns_a_copy(sample):
    out = de.drop_original_columns(sample)
    for col in ("Name", "Sex", "Embarked", "Cabin"):
        assert col not in out.columns
    assert "Name" in sample.columns          # el original no se tocó


# --- La cadena completa ----------------------------------------------------

def test_build_silver_runs_end_to_end(sample):
    """Con funciones de paso aún sin implementar, la cadena no debe romperse.

    Protege contra el error de escribir los cuerpos pendientes como '...',
    que devuelve None y rompe el primer .pipe().
    """
    out = de.build_silver(sample)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(sample)


def test_run_writes_both_parquets(tmp_path, sample):
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir()
    sample.to_csv(bronze_dir / "titanic_labeled.csv", index=False)

    out_dir = tmp_path / "silver"
    de.run(in_dir=bronze_dir, out_dir=out_dir)

    assert (out_dir / "titanic_silver.parquet").exists()
    assert (out_dir / "titanic_final.parquet").exists()

    completo = pd.read_parquet(out_dir / "titanic_silver.parquet")
    final = pd.read_parquet(out_dir / "titanic_final.parquet")
    assert "Name" in completo.columns        # Silver completo conserva todo
    assert "Name" not in final.columns       # la vista final, no
