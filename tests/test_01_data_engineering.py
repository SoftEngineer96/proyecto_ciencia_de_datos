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


# --- Etiqueta de supervivencia (Paso 16) -----------------------------------

def test_add_survivor_label(sample):
    out = de.add_survivor_label(sample)
    assert out["Survivor"].tolist() == ["No", "Yes", "Yes", "No", "Yes", "No"]


def test_survived_stays_numeric_so_its_mean_is_the_rate(sample):
    """La media de Survived es la tasa de supervivencia: no debe volverse texto."""
    out = de.add_survivor_label(sample)
    assert out["Survived"].dtype.kind in "iu"
    assert out["Survived"].mean() == pytest.approx(0.5)


# --- Títulos (Pasos 25 y 26) -----------------------------------------------

def test_titles_are_extracted_and_consolidated(sample):
    """Prueba del enunciado: Master -> Mr, the Countess -> Mrs, Dr -> Otro."""
    out = de.consolidate_titles(de.add_title(sample))
    assert out["TTitulo"].tolist() == ["Mr", "Mrs", "Miss", "Mr", "Mrs", "Otro"]


def test_add_title_extracts_the_raw_title(sample):
    """Antes de consolidar, el título es el que trae el nombre."""
    out = de.add_title(sample)
    assert out["TTitulo"].tolist() == [
        "Mr", "Mrs", "Miss", "Master", "the Countess", "Dr",
    ]


# --- Imputaciones (Pasos 21 y 22) ------------------------------------------

def test_impute_embarked_uses_mode(sample):
    """Prueba del enunciado."""
    out = de.impute_embarked(sample)
    assert out["Embarked"].isna().sum() == 0
    assert out.loc[3, "Embarked"] == "S"


def test_impute_age_by_sex_keeps_original(sample):
    """Prueba del enunciado."""
    out = de.impute_age(sample)
    assert out["Age"].isna().sum() == 0
    assert out.loc[2, "Age"] == pytest.approx(35.5)     # promedio de mujeres: (38 + 33) / 2
    assert out.loc[5, "Age"] == pytest.approx(12.0)     # promedio de hombres: (22 + 2) / 2
    assert out["AgeOriginal"].isna().sum() == 2
    assert out["AgeImputada"].tolist() == [False, False, True, False, False, True]


def test_impute_age_reduces_dispersion(sample):
    """Reto 22.1: imputar con la media añade valores al centro y baja la desviación."""
    out = de.impute_age(sample)
    assert out["Age"].std() < out["AgeOriginal"].std()


# --- Codificación (Pasos 21, 23, 24 y 26) ----------------------------------

def test_encode_categoricals_uses_fixed_maps(sample):
    entrada = de.consolidate_titles(de.add_title(de.add_cabin_features(de.impute_embarked(sample))))
    out = de.encode_categoricals(entrada)
    assert out["NSex"].tolist() == [1, 0, 0, 1, 0, 1]
    assert out["NEmbarked"].tolist() == [0, 1, 0, 0, 0, 2]
    assert out["NCabin"].tolist() == [8, 2, 8, 8, 1, 8]    # X=8, C=2, B=1 sobre DECKS
    assert out["NTitulo"].tolist() == [0, 1, 2, 0, 1, 3]


def test_ncabin_map_does_not_depend_on_the_data(sample):
    """El mapa sale de DECKS, no de las cubiertas presentes.

    Con sorted(df['TCabin'].unique()) estos dos subconjuntos darían números
    distintos para la misma cubierta, y los resultados dejarían de ser
    comparables entre ejecuciones.
    """
    def preparar(d):
        return de.encode_categoricals(
            de.consolidate_titles(de.add_title(de.add_cabin_features(de.impute_embarked(d))))
        )

    completo = preparar(sample)
    parcial = preparar(sample.iloc[[1, 4]])
    assert completo.loc[1, "NCabin"] == parcial.loc[1, "NCabin"] == 2    # cubierta C
    assert completo.loc[4, "NCabin"] == parcial.loc[4, "NCabin"] == 1    # cubierta B


# --- Contrato de salida (Paso 33) ------------------------------------------

def test_validate_silver_accepts_a_complete_pipeline(sample):
    out = de.build_silver(sample)
    assert de.validate_silver(out) is out


def test_validate_silver_rejects_nulls_in_derived_columns(sample):
    roto = de.build_silver(sample)
    roto.loc[0, "NSex"] = None
    with pytest.raises(ValueError):
        de.validate_silver(roto)


def test_validate_silver_allows_nulls_in_age_original(sample):
    """AgeOriginal conserva los NaN a proposito: son el registro de qué se imputó."""
    out = de.build_silver(sample)
    assert out["AgeOriginal"].isna().sum() == 2
    de.validate_silver(out)


# --- Pureza de todas las transformaciones ya implementadas -----------------

def test_functions_do_not_mutate_their_input(sample):
    """Prueba del enunciado: ninguna transformación altera lo que recibe."""
    before = sample.copy(deep=True)
    for fn in (de.add_person, de.add_family_features, de.add_cabin_features,
               de.cast_types, de.add_title, de.impute_embarked, de.impute_age):
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
