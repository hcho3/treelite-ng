"""Tests for XGBoost integration"""
# pylint: disable=R0201, R0915, R0913, R0914
import json
import os
import pathlib

import numpy as np
import pytest
import treelite
from hypothesis import given, settings
from hypothesis.strategies import data as hypothesis_callback
from hypothesis.strategies import integers, just, lists, sampled_from

from .hypothesis_util import (
    standard_classification_datasets,
    standard_multi_target_binary_classification_datasets,
    standard_regression_datasets,
    standard_settings,
)
from .metadata import dataset_db
from .util import TemporaryDirectory, load_txt

try:
    import xgboost as xgb
except ImportError:
    # skip this test suite if XGBoost is not installed
    pytest.skip("XGBoost not installed; skipping", allow_module_level=True)

try:
    from sklearn.datasets import load_svmlight_file
    from sklearn.model_selection import train_test_split
except ImportError:
    # skip this test suite if scikit-learn is not installed
    pytest.skip("scikit-learn not installed; skipping", allow_module_level=True)


def generate_data_for_squared_log_error(n_targets: int = 1):
    """Generate data containing outliers."""
    n_rows = 4096
    n_cols = 16

    outlier_mean = 10000  # mean of generated outliers
    n_outliers = 64

    X = np.random.randn(n_rows, n_cols)
    y = np.random.randn(n_rows, n_targets)
    y += np.abs(np.min(y, axis=0))

    # Create outliers
    for _ in range(0, n_outliers):
        ind = np.random.randint(0, len(y) - 1)
        y[ind, :] += np.random.randint(0, outlier_mean)

    # rmsle requires all label be greater than -1.
    assert np.all(y > -1.0)

    return X, y


@pytest.mark.parametrize(
    "objective",
    [
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:pseudohubererror",
    ],
)
@given(
    model_format=sampled_from(["legacy_binary", "json"]),
    pred_margin=sampled_from([True, False]),
    num_boost_round=integers(min_value=5, max_value=50),
    num_parallel_tree=integers(min_value=1, max_value=5),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_xgb_regressor(
    objective, model_format, pred_margin, num_boost_round, num_parallel_tree, callback
):
    # pylint: disable=too-many-locals
    """Test XGBoost with regression data"""

    # See https://github.com/dmlc/xgboost/pull/9574
    if objective == "reg:pseudohubererror":
        pytest.xfail("XGBoost 2.0 has a bug in the serialization of Pseudo-Huber error")

    if objective == "reg:squaredlogerror":
        X, y = generate_data_for_squared_log_error()
    else:
        X, y = callback.draw(standard_regression_datasets())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {
        "max_depth": 8,
        "eta": 1,
        "verbosity": 0,
        "objective": objective,
        "num_parallel_tree": num_parallel_tree,
    }
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "test")],
    )
    with TemporaryDirectory() as tmpdir:
        if model_format == "json":
            model_name = "model.json"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.frontend.load_xgboost_model(model_path)
        else:
            model_name = "model.model"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.frontend.load_xgboost_model_legacy_binary(model_path)
        assert (
            len(json.loads(tl_model.dump_as_json())["trees"])
            == num_boost_round * num_parallel_tree
        )

        out_pred = treelite.gtil.predict(tl_model, X_test, pred_margin=pred_margin)
        expected_pred = xgb_model.predict(dtest, output_margin=pred_margin).reshape(
            (X_test.shape[0], -1)
        )
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=3)


@given(
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=3, max_value=10), n_informative=just(5)
    ),
    objective=sampled_from(["multi:softmax", "multi:softprob"]),
    pred_margin=sampled_from([True, False]),
    model_format=sampled_from(["legacy_binary", "json"]),
    num_boost_round=integers(min_value=5, max_value=50),
    num_parallel_tree=integers(min_value=1, max_value=5),
)
@settings(**standard_settings())
def test_xgb_multiclass_classifier(
    dataset, objective, pred_margin, model_format, num_boost_round, num_parallel_tree
):
    # pylint: disable=too-many-locals
    """Test XGBoost with Iris data (multi-class classification)"""
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_class = np.max(y) + 1
    param = {
        "max_depth": 6,
        "eta": 0.05,
        "num_class": num_class,
        "verbosity": 0,
        "objective": objective,
        "metric": "mlogloss",
        "num_parallel_tree": num_parallel_tree,
    }
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "test")],
    )

    with TemporaryDirectory() as tmpdir:
        if model_format == "json":
            model_name = "iris.json"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.frontend.load_xgboost_model(model_path)
        else:
            model_name = "iris.model"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.frontend.load_xgboost_model_legacy_binary(model_path)
        expected_num_tree = num_class * num_boost_round * num_parallel_tree
        assert len(json.loads(tl_model.dump_as_json())["trees"]) == expected_num_tree

        if objective == "multi:softmax" and not pred_margin:
            out_pred = treelite.gtil.predict_leaf(tl_model, X_test)
            expected_pred = xgb_model.predict(dtest, pred_leaf=True)
        else:
            out_pred = treelite.gtil.predict(tl_model, X_test, pred_margin=pred_margin)
            expected_pred = xgb_model.predict(dtest, output_margin=pred_margin)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@given(
    objective_pair=sampled_from(
        [
            ("binary:logistic", 2),
            ("binary:hinge", 2),
            ("binary:logitraw", 2),
            ("count:poisson", 4),
            ("rank:pairwise", 5),
            ("rank:ndcg", 5),
            ("rank:map", 2),
        ],
    ),
    pred_margin=sampled_from([True, False]),
    model_format=sampled_from(["legacy_binary", "json"]),
    num_boost_round=integers(min_value=5, max_value=50),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_xgb_nonlinear_objective(
    objective_pair, pred_margin, model_format, num_boost_round, callback
):
    # pylint: disable=too-many-locals
    """Test XGBoost with non-linear objectives with synthetic data"""
    objective, num_class = objective_pair
    X, y = callback.draw(
        standard_classification_datasets(
            n_classes=just(num_class), n_informative=just(5)
        )
    )
    assert np.min(y) == 0
    assert np.max(y) == num_class - 1

    dtrain = xgb.DMatrix(X, label=y)
    if objective.startswith("rank:"):
        dtrain.set_group([X.shape[0]])
    xgb_model = xgb.train(
        {"objective": objective, "base_score": 0.5, "seed": 0},
        dtrain=dtrain,
        num_boost_round=num_boost_round,
    )

    objective_tag = objective.replace(":", "_")
    if model_format == "json":
        model_name = f"nonlinear_{objective_tag}.json"
    else:
        model_name = f"nonlinear_{objective_tag}.bin"
    with TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, model_name)
        xgb_model.save_model(model_path)

        if model_format == "json":
            tl_model = treelite.frontend.load_xgboost_model(model_path)
        else:
            tl_model = treelite.frontend.load_xgboost_model_legacy_binary(model_path)

        out_pred = treelite.gtil.predict(tl_model, X, pred_margin=pred_margin)
        expected_pred = xgb_model.predict(dtrain, output_margin=pred_margin).reshape(
            (X.shape[0], -1)
        )
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize("in_memory", [True, False])
def test_xgb_categorical_split(in_memory):
    """Test toy XGBoost model with categorical splits"""
    dataset = "xgb_toy_categorical"
    if in_memory:
        xgb_model = xgb.Booster(model_file=dataset_db[dataset].model)
        tl_model = treelite.Model.from_xgboost(xgb_model)
    else:
        tl_model = treelite.frontend.load_xgboost_model(dataset_db[dataset].model)

    X, _ = load_svmlight_file(dataset_db[dataset].dtest, zero_based=True)
    expected_pred = load_txt(dataset_db[dataset].expected_margin).reshape(
        (X.shape[0], -1)
    )
    out_pred = treelite.gtil.predict(tl_model, X.toarray(), pred_margin=True)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@given(
    dataset=standard_classification_datasets(n_classes=just(2)),
    model_format=sampled_from(["legacy_binary", "json"]),
    num_boost_round=integers(min_value=5, max_value=50),
)
@settings(**standard_settings())
def test_xgb_dart(dataset, model_format, num_boost_round):
    # pylint: disable=too-many-locals
    """Test XGBoost DART model with dummy data"""
    X, y = dataset

    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "booster": "dart",
        "max_depth": 5,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "sample_type": "uniform",
        "normalize_type": "tree",
        "rate_drop": 0.1,
        "skip_drop": 0.5,
    }
    xgb_model = xgb.train(param, dtrain=dtrain, num_boost_round=num_boost_round)

    with TemporaryDirectory() as tmpdir:
        if model_format == "json":
            model_name = "dart.json"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.frontend.load_xgboost_model(model_path)
        else:
            tl_model = treelite.Model.from_xgboost(xgb_model)
        out_pred = treelite.gtil.predict(tl_model, X, pred_margin=True)
        expected_pred = xgb_model.predict(dtrain, output_margin=True).reshape(
            (X.shape[0], -1)
        )
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@given(
    random_integer_seq=lists(
        integers(min_value=0, max_value=20), min_size=1, max_size=10
    ),
    extra_field_type=sampled_from(["string", "object", "list"]),
    use_tempfile=sampled_from([True, False]),
)
@settings(print_blob=True, deadline=None)
def test_extra_field_in_xgb_json(random_integer_seq, extra_field_type, use_tempfile):
    # pylint: disable=too-many-locals,too-many-arguments
    """
    Test if we can handle extra fields in XGBoost JSON model file
    Insert an extra field at a random place and then load the model into Treelite,
    as follows:
    * Use the Hypothesis package to generate a random sequence of integers.
    * Then use the integer sequence to navigate through the XGBoost model JSON object.
    * Lastly, after navigating, insert the extra field at that location.
    """
    np.random.seed(0)
    nrow = 16
    ncol = 8
    X = np.random.randn(nrow, ncol)
    y = np.random.randint(0, 2, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == 1

    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "max_depth": 1,
        "eta": 1,
        "objective": "binary:logistic",
        "verbosity": 0,
    }
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=1,
    )
    model_json_str = bst.save_raw(raw_format="json").decode(encoding="utf-8")
    model_obj = json.loads(model_json_str)

    def get_extra_field_value():
        if extra_field_type == "object":
            return {}
        if extra_field_type == "string":
            return "extra"
        if extra_field_type == "list":
            return []
        return None

    def insert_extra_field(model_obj, seq):
        if (not seq) or (not model_obj):
            if isinstance(model_obj, dict):
                model_obj["extra_field"] = get_extra_field_value()
                return True
        elif isinstance(model_obj, list):
            idx = seq[0] % len(model_obj)
            subobj = model_obj[idx]
            if isinstance(subobj, (dict, list)):
                return insert_extra_field(subobj, seq[1:])
        elif isinstance(model_obj, dict):
            idx = seq[0] % len(model_obj)
            subobj = list(model_obj.items())[idx][1]
            if isinstance(subobj, (dict, list)):
                if not insert_extra_field(subobj, seq[1:]):
                    model_obj["extra_field"] = get_extra_field_value()
                return True
            model_obj["extra_field"] = get_extra_field_value()
            return True
        return False

    insert_extra_field(model_obj, random_integer_seq)
    new_model_str = json.dumps(model_obj)
    assert "extra_field" in new_model_str
    if use_tempfile:
        with TemporaryDirectory() as tmpdir:
            new_model_path = os.path.join(tmpdir, "new_model.json")
            with open(new_model_path, "w", encoding="utf-8") as f:
                f.write(new_model_str)
            treelite.frontend.load_xgboost_model(
                new_model_path, allow_unknown_field=True
            )
    else:
        treelite.frontend.from_xgboost_json(new_model_str, allow_unknown_field=True)


@given(
    dataset=standard_multi_target_binary_classification_datasets(
        n_targets=integers(min_value=2, max_value=10)
    ),
    num_boost_round=integers(min_value=5, max_value=50),
    num_parallel_tree=integers(min_value=1, max_value=3),
    multi_strategy=sampled_from(["one_output_per_tree", "multi_output_tree"]),
    pred_margin=sampled_from([True, False]),
    in_memory=sampled_from([True, False]),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_xgb_multi_target_binary_classifier(
    dataset,
    num_boost_round,
    num_parallel_tree,
    multi_strategy,
    pred_margin,
    in_memory,
    callback,
):
    """Test XGBoost with multi-target classification problem"""
    X, y = dataset
    if multi_strategy == "one_output_per_tree" and not in_memory:
        model_format = callback.draw(sampled_from(["legacy_binary", "json"]))
    else:
        model_format = callback.draw(just("json"))

    clf = xgb.XGBClassifier(
        n_estimators=num_boost_round,
        tree_method="hist",
        learning_rate=0.1,
        max_depth=8,
        objective="binary:logistic",
        num_parallel_tree=num_parallel_tree,
        multi_strategy=multi_strategy,
    )
    clf.fit(X, y)

    if in_memory:
        tl_model = treelite.frontend.from_xgboost(clf.get_booster())
    else:
        with TemporaryDirectory() as tmpdir:
            if model_format == "json":
                model_path = pathlib.Path(tmpdir) / "multi_target.json"
                clf.save_model(model_path)
                tl_model = treelite.frontend.load_xgboost_model(model_path)
            else:
                model_path = pathlib.Path(tmpdir) / "multi_target.model"
                clf.save_model(model_path)
                tl_model = treelite.frontend.load_xgboost_model_legacy_binary(
                    model_path
                )

    out_pred = treelite.gtil.predict(tl_model, X, pred_margin=pred_margin)
    if pred_margin:
        expected_pred = clf.predict(X, output_margin=True)
    else:
        expected_pred = clf.predict_proba(X)
    expected_pred = np.transpose(expected_pred[:, :, np.newaxis], axes=(1, 0, 2))
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize(
    "objective",
    [
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:pseudohubererror",
    ],
)
@given(
    n_targets=integers(min_value=2, max_value=6),
    pred_margin=sampled_from([True, False]),
    num_boost_round=integers(min_value=5, max_value=10),
    num_parallel_tree=integers(min_value=1, max_value=3),
    multi_strategy=sampled_from(["one_output_per_tree", "multi_output_tree"]),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_xgb_multi_target_regressor(
    n_targets,
    objective,
    pred_margin,
    num_boost_round,
    num_parallel_tree,
    multi_strategy,
    callback,
):
    # pylint: disable=too-many-locals
    """Test XGBoost with regression data"""

    # See https://github.com/dmlc/xgboost/pull/9574
    if objective == "reg:pseudohubererror":
        pytest.xfail("XGBoost 2.0 has a bug in the serialization of Pseudo-Huber error")

    if objective == "reg:squaredlogerror":
        X, y = generate_data_for_squared_log_error(n_targets=n_targets)
    else:
        X, y = callback.draw(standard_regression_datasets(n_targets=just(n_targets)))
    if multi_strategy == "one_output_per_tree":
        model_format = callback.draw(sampled_from(["legacy_binary", "json"]))
    else:
        model_format = callback.draw(just("json"))
    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "max_depth": 8,
        "eta": 0.1,
        "verbosity": 0,
        "objective": objective,
        "num_parallel_tree": num_parallel_tree,
        "multi_strategy": multi_strategy,
    }
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_boost_round,
    )

    with TemporaryDirectory() as tmpdir:
        if model_format == "json":
            model_name = "model.json"
            model_path = pathlib.Path(tmpdir) / model_name
            xgb_model.save_model(model_path)
            tl_model = treelite.frontend.load_xgboost_model(model_path)
        else:
            model_name = "model.model"
            model_path = pathlib.Path(tmpdir) / model_name
            xgb_model.save_model(model_path)
            tl_model = treelite.frontend.load_xgboost_model_legacy_binary(model_path)
        expected_n_trees = num_boost_round * num_parallel_tree
        if multi_strategy == "one_output_per_tree":
            expected_n_trees *= n_targets
        assert len(json.loads(tl_model.dump_as_json())["trees"]) == expected_n_trees

        out_pred = treelite.gtil.predict(tl_model, X, pred_margin=pred_margin)
        expected_pred = xgb_model.predict(dtrain, output_margin=pred_margin)
        expected_pred = np.transpose(expected_pred[:, :, np.newaxis], axes=(1, 0, 2))
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=3)
