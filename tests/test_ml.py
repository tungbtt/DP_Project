import json
import zipfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dataprep.cli import main
from dataprep.errors import DataPrepError
from dataprep.ml import MLConfig, export_ml_bundle, prepare_ml


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "id": [f"{i:03}" for i in range(60)],
            "value": np.arange(60, dtype=float),
            "city": ["A", "B", None] * 20,
            "target": np.arange(60, dtype=float) * 2,
            "label": ["yes", "no", "maybe"] * 20,
            "day": pd.date_range("2026-01-01", periods=60).strftime("%Y-%m-%d"),
        }
    )


@pytest.fixture
def cfg():
    return MLConfig(target="target", numeric_columns=["value"], categorical_columns=["city"])


def test_train_only_fit_and_unseen_categories(data, cfg):
    first = prepare_ml(data, cfg)
    train, held_out = first.row_indices["train"], first.row_indices["test"]
    original = data.copy(deep=True)
    shifted = data.copy(deep=True)
    shifted.loc[held_out, "value"] = 1e9
    shifted.loc[held_out, "city"] = "ONLY_TEST"
    result = prepare_ml(shifted, cfg)
    scale = result.preprocessor.named_steps["columns"].named_transformers_["numeric"].named_steps["scaler"]
    assert scale.mean_[0] == pytest.approx(data.loc[train, "value"].mean())
    assert "ONLY_TEST" not in " ".join(result.feature_names)
    np.testing.assert_allclose(first.matrices["train"].toarray(), result.matrices["train"].toarray())
    assert result.matrices["test"][:, 1:].nnz == 0
    assert_frame_equal(data, original)


def test_imputation_learns_only_train_and_aligns_targets(data, cfg):
    baseline = prepare_ml(data, cfg)
    train = baseline.row_indices["train"]
    data.loc[train[:5], "value"] = np.nan
    data.loc[baseline.row_indices["test"], "value"] = 99999
    result = prepare_ml(data, cfg)
    imputer = result.preprocessor.named_steps["columns"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == pytest.approx(data.loc[train, "value"].median())
    assert result.matrices["train"][:, 0].toarray().mean() == pytest.approx(0, abs=1e-10)
    sets = [set(rows) for rows in result.row_indices.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(range(len(data)))
    for name, rows in result.row_indices.items():
        np.testing.assert_array_equal(result.targets[name], data.loc[rows, "target"])
        assert result.matrices[name].shape[1] == len(result.feature_names)


@pytest.mark.parametrize("scaler", ["standard", "minmax", "robust", "none"])
def test_scaling_contracts(data, cfg, scaler):
    config = replace(cfg, scaler=scaler, categorical_columns=[])
    result = prepare_ml(data, config)
    values = result.matrices["train"].toarray().ravel()
    if scaler == "standard":
        assert values.mean() == pytest.approx(0, abs=1e-10)
        assert values.std() == pytest.approx(1)
    elif scaler == "minmax":
        assert values.min() == pytest.approx(0)
        assert values.max() == pytest.approx(1)
    elif scaler == "robust":
        assert np.median(values) == pytest.approx(0, abs=1e-10)
        assert np.quantile(values, 0.75) - np.quantile(values, 0.25) == pytest.approx(1)
    else:
        np.testing.assert_array_equal(values, data.loc[result.row_indices["train"], "value"])


def test_constant_empty_features_and_invalid_numeric(data, cfg):
    data["value"] = np.nan
    data["city"] = None
    data["constant"] = 5.0
    result = prepare_ml(data, replace(cfg, numeric_columns=["value", "constant"]))
    assert result.matrices["train"].shape[1] == 3
    assert np.isfinite(result.matrices["train"].data).all()
    assert "value" in result.manifest["all_missing_train_columns"]
    assert "city" in result.manifest["all_missing_train_columns"]
    data["value"] = ["bad"] + ["2"] * 59
    with pytest.raises(DataPrepError, match="không phải số"):
        prepare_ml(data, cfg)
    result = prepare_ml(data, replace(cfg, numeric_errors="coerce"))
    assert result.manifest["invalid_numeric_to_missing"]["value"] == 1


def test_stratified_classification_label_mapping(data, cfg):
    result = prepare_ml(data, replace(cfg, task="classification", target="label"))
    assert set(result.manifest["target_mapping"]) == {"yes", "no", "maybe"}
    for split, target in result.targets.items():
        assert set(target) == {0, 1, 2}
        labels = result.target_encoder.inverse_transform(target)
        np.testing.assert_array_equal(labels, data.loc[result.row_indices[split], "label"])


def test_time_split_and_repeated_timestamp_boundaries(data, cfg):
    config = replace(cfg, split_method="time", time_column="day", time_format="%Y-%m-%d")
    result = prepare_ml(data, config)
    assert result.row_indices["train"].max() < result.row_indices["validation"].min()
    assert result.row_indices["validation"].max() < result.row_indices["test"].min()
    data["day"] = "2026-01-01"
    with pytest.raises(DataPrepError, match="timestamp"):
        prepare_ml(data, config)


def test_missing_target_is_explicit_and_rows_retained(data, cfg):
    data.loc[3, "target"] = np.nan
    with pytest.raises(DataPrepError, match="thiếu target"):
        prepare_ml(data, cfg)
    result = prepare_ml(data, replace(cfg, drop_missing_target=True, validation_size=0))
    assert result.manifest["excluded_target_rows"] == [3]
    assert len(result.row_indices["validation"]) == 0
    assert result.matrices["validation"].shape == (0, len(result.feature_names))
    assert 3 not in set.union(*(set(v) for v in result.row_indices.values()))


def test_minmax_unseen_values_not_clipped_and_ordinal_unknown(data, cfg):
    baseline = prepare_ml(data, cfg)
    data.loc[baseline.row_indices["test"], "value"] = 10000.0
    data.loc[baseline.row_indices["test"], "city"] = "NEW"
    result = prepare_ml(data, replace(cfg, scaler="minmax", encoding="ordinal"))
    assert (result.matrices["test"][:, 0].toarray() > 1).all()
    assert (result.matrices["test"][:, 1].toarray() == -1).all()


def test_bundle_and_reusable_fitted_transformer(data, cfg):
    result = prepare_ml(data, cfg)
    with zipfile.ZipFile(BytesIO(export_ml_bundle(result, max_csv_cells=0))) as archive:
        assert "X_train.npz" in archive.namelist()
        assert "X_train.csv" not in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["fit_partition"] == "train"
        fitted = joblib.load(BytesIO(archive.read("preprocessor.joblib")))
        actual = fitted.transform(data.loc[result.row_indices["test"]])
        np.testing.assert_allclose(actual.toarray(), result.matrices["test"].toarray())
        with pytest.raises(DataPrepError, match="Thiếu feature"):
            fitted.transform(pd.DataFrame({"wrong": [1]}))


@pytest.mark.parametrize(
    "updates",
    [
        {"numeric_columns": ["target"]},
        {"numeric_columns": [], "categorical_columns": []},
        {"numeric_columns": ["value"], "categorical_columns": ["value"]},
        {"test_size": 0.5},
        {"test_size": 0.45, "validation_size": 0.4},
        {"numeric_columns": ["absent"]},
        {"scaler": "unknown"},
        {"max_categories": 1},
        {"seed": -1},
        {"stratify": "yes"},
    ],
)
def test_invalid_configs(data, cfg, updates):
    with pytest.raises(DataPrepError):
        prepare_ml(data, replace(cfg, **updates))


def test_singleton_class_rejected_instead_of_silently_disabling_stratification(data, cfg):
    data["label"] = ["rare"] + ["common"] * 59
    with pytest.raises(DataPrepError, match="Không chia được"):
        prepare_ml(data, replace(cfg, target="label", task="classification"))


def test_duplicate_overlap_is_reported(data, cfg):
    data["value"], data["city"], data["target"] = 1.0, "same", 2.0
    result = prepare_ml(data, cfg)
    assert result.manifest["duplicate_signatures_across_splits"] == 1


def test_cli_ml_mode(tmp_path):
    root = Path(__file__).resolve().parents[1]
    args = [
        str(root / "examples/sales_dirty.csv"),
        "--ml-config",
        str(root / "examples/ml_config.json"),
        "--output",
        str(tmp_path),
    ]
    assert main(args) == 0
    assert (tmp_path / "ml_ready.zip").is_file()
    assert json.loads((tmp_path / "ml_manifest.json").read_text(encoding="utf-8"))["fit_partition"] == "train"
    assert main(args) == 2
    assert main(args + ["--pipeline", str(root / "examples/pipeline.json"), "--overwrite"]) == 2


def test_config_json_rejects_unknown_field():
    with pytest.raises(DataPrepError):
        MLConfig.from_json('{"target":"y","typo":true}')
