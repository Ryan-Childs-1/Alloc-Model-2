from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Force standalone Keras to use the PyTorch backend when available. This avoids TensorFlow and JAX installs.
os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from allocation_engine import _raw_prediction_from_heuristic
from feature_engineering import build_model_frame

ROOT = Path(".")
PREPROCESSOR_PATH = ROOT / "preprocessor.joblib"
KERAS_MODEL_PATH = ROOT / "allocation_model_keras.keras"
FALLBACK_MODEL_PATH = ROOT / "allocation_fallback_ensemble.joblib"
META_PATH = ROOT / "model_metadata.json"


def _split_cols(X: pd.DataFrame):
    feature_cols = [c for c in X.columns if c != "__excel_row"]
    cat_cols = [c for c in feature_cols if c.startswith("cat__")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return num_cols, cat_cols


def _make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse=False)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = _split_cols(X)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("onehot", _make_onehot()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def _to_dense_float32(x) -> np.ndarray:
    arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return np.asarray(arr, dtype=np.float32)


def _try_keras():
    try:
        os.environ.setdefault("KERAS_BACKEND", "torch")
        import keras
        return keras
    except Exception:
        return None


def _build_keras_model(input_dim: int):
    keras = _try_keras()
    if keras is None:
        raise RuntimeError("Keras could not be imported. Install keras and torch, or use the fallback ensemble.")
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.Dense(512, activation="swish", kernel_regularizer=keras.regularizers.l2(1e-5))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.18)(x)
    x = keras.layers.Dense(256, activation="swish", kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.14)(x)
    x = keras.layers.Dense(128, activation="swish")(x)
    x = keras.layers.Dense(64, activation="swish")(x)
    alloc_prob = keras.layers.Dense(1, activation="sigmoid", name="alloc_prob")(x)
    log_qty = keras.layers.Dense(1, activation="softplus", name="log_qty")(x)
    model = keras.Model(inputs=inputs, outputs={"alloc_prob": alloc_prob, "log_qty": log_qty})
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.0012, weight_decay=1e-5),
        loss={"alloc_prob": "binary_crossentropy", "log_qty": keras.losses.Huber(delta=0.75)},
        loss_weights={"alloc_prob": 1.75, "log_qty": 1.0},
        metrics={"alloc_prob": ["accuracy"]},
    )
    return model


def _train_keras_model(X_train, y_cls_train, y_qty_train, X_val, y_cls_val, y_qty_val, model_dir: Path) -> Dict[str, Any]:
    keras = _try_keras()
    if keras is None:
        raise RuntimeError("Keras is unavailable.")
    model = _build_keras_model(int(X_train.shape[1]))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, min_delta=1e-4),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=2e-5),
    ]
    y_train = {
        "alloc_prob": y_cls_train.astype("float32").reshape(-1, 1),
        "log_qty": np.log1p(y_qty_train.astype("float32")).reshape(-1, 1),
    }
    y_val = {
        "alloc_prob": y_cls_val.astype("float32").reshape(-1, 1),
        "log_qty": np.log1p(y_qty_val.astype("float32")).reshape(-1, 1),
    }
    # Weight positive allocation rows more heavily so the model learns real allocation behavior without over-allocating blanks.
    pos_rate = float(np.mean(y_cls_train)) if len(y_cls_train) else 0.0
    pos_weight = max(1.0, min(7.0, (1.0 - pos_rate) / max(pos_rate, 1e-4)))
    sw_cls = np.where(y_cls_train > 0, pos_weight, 1.0).astype("float32")
    sw_qty = (0.35 + 1.65 * y_cls_train.astype("float32")).astype("float32")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=min(512, max(32, len(X_train))),
        verbose=0,
        callbacks=callbacks,
        sample_weight={"alloc_prob": sw_cls, "log_qty": sw_qty},
    )
    model.save(model_dir / KERAS_MODEL_PATH.name)
    pred = model.predict(X_val, verbose=0)
    prob = np.asarray(pred["alloc_prob"]).reshape(-1)
    qty = np.expm1(np.asarray(pred["log_qty"]).reshape(-1)).clip(min=0)
    yhat = (prob >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_cls_val, yhat, average="binary", zero_division=0)
    return {
        "backend": "keras_torch_mlp",
        "keras_backend": os.environ.get("KERAS_BACKEND", "torch"),
        "input_dim": int(X_train.shape[1]),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "val_mae_raw": float(mean_absolute_error(y_qty_val, qty)),
        "val_precision": float(pr),
        "val_recall": float(rc),
        "val_f1": float(f1),
        "best_val_loss": float(min(history.history.get("val_loss", [np.nan]))),
    }


def _keras_predict(model_path: Path, X_dense: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    keras = _try_keras()
    if keras is None:
        raise RuntimeError("Keras model exists, but Keras/torch is not installed.")
    model = keras.saving.load_model(model_path)
    pred = model.predict(X_dense, verbose=0)
    prob = np.asarray(pred["alloc_prob"]).reshape(-1)
    qty = np.expm1(np.asarray(pred["log_qty"]).reshape(-1)).clip(min=0)
    return qty, prob


def _train_fallback_ensemble(Xt_dense, y_reg_train, y_cls_train, Xv_dense, y_reg_val, y_cls_val) -> Dict[str, Any]:
    classifiers = {
        "hgb": HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, l2_regularization=0.02, random_state=42),
        "gb": GradientBoostingClassifier(n_estimators=180, learning_rate=0.045, max_depth=3, random_state=43),
        "rf": RandomForestClassifier(n_estimators=140, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=44),
        "et": ExtraTreesClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=45),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=0.001, batch_size="auto", learning_rate_init=0.001, max_iter=120, early_stopping=True, random_state=46),
    }
    fitted_clf = {}
    for name, clf in classifiers.items():
        try:
            clf.fit(Xt_dense, y_cls_train)
            fitted_clf[name] = clf
        except Exception:
            pass

    y_log = np.log1p(np.asarray(y_reg_train, dtype=float))
    regressors = {
        "hgb": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.045, l2_regularization=0.02, loss="absolute_error", random_state=47),
        "gb": GradientBoostingRegressor(n_estimators=200, learning_rate=0.045, max_depth=3, loss="huber", random_state=48),
        "rf": RandomForestRegressor(n_estimators=140, min_samples_leaf=2, n_jobs=-1, random_state=49),
        "et": ExtraTreesRegressor(n_estimators=180, min_samples_leaf=2, n_jobs=-1, random_state=50),
        "mlp": MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=0.001, learning_rate_init=0.001, max_iter=140, early_stopping=True, random_state=51),
    }
    fitted_reg = {}
    for name, reg in regressors.items():
        try:
            reg.fit(Xt_dense, y_log)
            fitted_reg[name] = reg
        except Exception:
            pass

    probs = []
    for clf in fitted_clf.values():
        if hasattr(clf, "predict_proba"):
            probs.append(clf.predict_proba(Xv_dense)[:, 1])
        else:
            probs.append(np.asarray(clf.predict(Xv_dense), dtype=float))
    prob = np.mean(probs, axis=0) if probs else np.zeros(len(Xv_dense))

    raws = []
    for reg in fitted_reg.values():
        raws.append(np.expm1(reg.predict(Xv_dense)).clip(min=0))
    raw = np.mean(raws, axis=0) if raws else np.zeros(len(Xv_dense))
    yhat = (prob >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_cls_val, yhat, average="binary", zero_division=0)
    return {
        "backend": "sklearn_ensemble_plus_mlp",
        "classifiers": fitted_clf,
        "regressors": fitted_reg,
        "val_mae_raw": float(mean_absolute_error(y_reg_val, raw)),
        "fallback_precision": float(pr),
        "fallback_recall": float(rc),
        "fallback_f1": float(f1),
    }


def _fallback_predict(model: Dict[str, Any], X_dense: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    for clf in model.get("classifiers", {}).values():
        if hasattr(clf, "predict_proba"):
            probs.append(clf.predict_proba(X_dense)[:, 1])
        else:
            probs.append(np.asarray(clf.predict(X_dense), dtype=float))
    prob = np.mean(probs, axis=0) if probs else np.zeros(len(X_dense))
    raws = []
    for reg in model.get("regressors", {}).values():
        raws.append(np.expm1(np.clip(reg.predict(X_dense), 0, 8)).clip(min=0))
    raw = np.mean(raws, axis=0) if raws else np.zeros(len(X_dense))
    return raw, prob


def train_models_from_frames(frames: List[pd.DataFrame], schema_path: str = "feature_schema.json", model_dir: str | Path = ".") -> Dict[str, Any]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    X_list, y_reg_list, y_cls_list = [], [], []
    for df in frames:
        X, y_reg, y_cls, _ = build_model_frame(df, schema_path=schema_path, training=True)
        if y_reg is not None and len(X):
            X_list.append(X)
            y_reg_list.append(y_reg)
            y_cls_list.append(y_cls)
    if not X_list:
        raise ValueError("No training rows were found. Check that Final Alloc. is present and the workbook schema matches.")
    X = pd.concat(X_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    y_reg = pd.concat(y_reg_list, ignore_index=True).astype(float).clip(lower=0)
    y_cls = pd.concat(y_cls_list, ignore_index=True).astype(int)

    if len(X) < 20:
        raise ValueError("Not enough rows to train a model. Provide more historical workbooks.")
    stratify = y_cls if y_cls.nunique() > 1 and y_cls.value_counts().min() >= 2 else None
    X_train, X_val, yr_train, yr_val, yc_train, yc_val = train_test_split(
        X, y_reg.values, y_cls.values, test_size=0.20, random_state=42, stratify=stratify
    )
    pre = make_preprocessor(X_train)
    Xt = _to_dense_float32(pre.fit_transform(X_train))
    Xv = _to_dense_float32(pre.transform(X_val))
    joblib.dump(pre, model_dir / PREPROCESSOR_PATH.name)

    fallback = _train_fallback_ensemble(Xt, yr_train, yc_train, Xv, yr_val, yc_val)
    joblib.dump(fallback, model_dir / FALLBACK_MODEL_PATH.name)

    keras_meta: Dict[str, Any] = {"backend": "unavailable", "keras_trained": False}
    primary = "sklearn_ensemble_plus_mlp"
    try:
        keras_meta = _train_keras_model(Xt, yc_train, yr_train, Xv, yc_val, yr_val, model_dir)
        keras_meta["keras_trained"] = True
        primary = "keras_torch_mlp"
    except Exception as exc:
        keras_meta = {"backend": "keras_unavailable", "keras_trained": False, "keras_error": str(exc)[:500]}

    meta = {
        "project": "Allocation AI v2",
        "primary_backend": primary,
        "rows_trained": int(len(X)),
        "positive_rows": int(np.sum(y_cls.values > 0)),
        "features_after_preprocessing": int(Xt.shape[1]),
        "keras": keras_meta,
        "fallback": {k: v for k, v in fallback.items() if k not in {"classifiers", "regressors"}},
        "artifact_layout": "flat",
    }
    with open(model_dir / META_PATH.name, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


class AllocationPredictor:
    def __init__(self, model_dir: str | Path = "."):
        self.model_dir = Path(model_dir)
        self.preprocessor = None
        self.keras_path = self.model_dir / KERAS_MODEL_PATH.name
        self.fallback = None
        pre_path = self.model_dir / PREPROCESSOR_PATH.name
        if pre_path.exists():
            self.preprocessor = joblib.load(pre_path)
        fb_path = self.model_dir / FALLBACK_MODEL_PATH.name
        if fb_path.exists():
            self.fallback = joblib.load(fb_path)

    def predict(self, df: pd.DataFrame, schema_path: str = "feature_schema.json") -> Tuple[np.ndarray, np.ndarray, str]:
        X, _, _, _ = build_model_frame(df, schema_path=schema_path, training=False)
        if self.preprocessor is None:
            # Full rule-backed fallback if no artifacts exist yet.
            from feature_engineering import canonicalize_dataframe, add_formula_features
            canon, _ = canonicalize_dataframe(df, schema_path)
            feat = add_formula_features(canon)
            raw, prob = _raw_prediction_from_heuristic(feat)
            return raw, prob, "rule_fallback_no_model_artifacts"
        Xt = _to_dense_float32(self.preprocessor.transform(X.replace([np.inf, -np.inf], np.nan)))
        if self.keras_path.exists():
            try:
                raw, prob = _keras_predict(self.keras_path, Xt)
                return raw, prob, "keras_torch_mlp"
            except Exception:
                pass
        if self.fallback is not None:
            raw, prob = _fallback_predict(self.fallback, Xt)
            return raw, prob, "sklearn_sgd_full_data_calibrated"
        from feature_engineering import canonicalize_dataframe, add_formula_features
        canon, _ = canonicalize_dataframe(df, schema_path)
        feat = add_formula_features(canon)
        raw, prob = _raw_prediction_from_heuristic(feat)
        return raw, prob, "rule_fallback_no_available_model"


def read_metadata(model_dir: str | Path = ".") -> Dict[str, Any]:
    path = Path(model_dir) / META_PATH.name
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
