from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import build_model_frame

MODEL_DIR = Path("models")
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
SKLEARN_MODEL_PATH = MODEL_DIR / "allocation_fallback_model.joblib"
KERAS_MODEL_PATH = MODEL_DIR / "allocation_model.keras"
META_PATH = MODEL_DIR / "model_metadata.json"


def _split_cols(X: pd.DataFrame):
    feature_cols = [c for c in X.columns if c != "__excel_row"]
    cat_cols = [c for c in feature_cols if c.startswith("cat__")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return num_cols, cat_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = _split_cols(X)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")


def _try_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except Exception:
        return None


def train_models_from_frames(frames: list[pd.DataFrame], schema_path: str = "feature_schema.json", model_dir: str | Path = MODEL_DIR) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    Xs, y_regs, y_clss = [], [], []
    for df in frames:
        X, y_reg, y_cls, _ = build_model_frame(df, schema_path=schema_path, training=True)
        Xs.append(X)
        y_regs.append(y_reg)
        y_clss.append(y_cls)
    X = pd.concat(Xs, ignore_index=True)
    y_reg = pd.concat(y_regs, ignore_index=True).fillna(0).clip(lower=0)
    y_cls = pd.concat(y_clss, ignore_index=True).fillna(0).astype(int)

    # Drop rows that are entirely malformed, but keep blanks/zeros as valid no-allocation targets.
    keep = X.drop(columns=["__excel_row"], errors="ignore").notna().any(axis=1)
    X = X.loc[keep].reset_index(drop=True)
    y_reg = y_reg.loc[keep].reset_index(drop=True)
    y_cls = y_cls.loc[keep].reset_index(drop=True)

    pre = make_preprocessor(X)
    X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
        X, y_reg, y_cls, test_size=0.18, random_state=42, stratify=y_cls if y_cls.nunique() == 2 and y_cls.value_counts().min() >= 5 else None
    )
    Xt = pre.fit_transform(X_train.drop(columns=["__excel_row"], errors="ignore"))
    Xv = pre.transform(X_val.drop(columns=["__excel_row"], errors="ignore"))
    joblib.dump(pre, model_dir / PREPROCESSOR_PATH.name)

    tf = _try_tensorflow()
    keras_trained = False
    keras_metrics: Dict[str, Any] = {}
    if tf is not None:
        # Sparse one-hot output can be fed to Keras as dense for moderate workbook-scale data.
        Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        Xv_dense = Xv.toarray() if hasattr(Xv, "toarray") else np.asarray(Xv)
        inputs = tf.keras.Input(shape=(Xt_dense.shape[1],), name="features")
        x = tf.keras.layers.Dense(384, activation="swish")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(192, activation="swish")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.20)(x)
        x = tf.keras.layers.Dense(96, activation="swish")(x)
        x = tf.keras.layers.Dropout(0.12)(x)
        cls = tf.keras.layers.Dense(1, activation="sigmoid", name="alloc_probability")(x)
        qty = tf.keras.layers.Dense(1, activation="softplus", name="raw_allocation")(x)
        model = tf.keras.Model(inputs=inputs, outputs=[cls, qty])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={"alloc_probability": "binary_crossentropy", "raw_allocation": tf.keras.losses.Huber(delta=8.0)},
            loss_weights={"alloc_probability": 1.6, "raw_allocation": 1.0},
            metrics={"alloc_probability": ["accuracy"], "raw_allocation": ["mae"]},
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        ]
        model.fit(
            Xt_dense,
            {"alloc_probability": y_cls_train.values, "raw_allocation": y_reg_train.values},
            validation_data=(Xv_dense, {"alloc_probability": y_cls_val.values, "raw_allocation": y_reg_val.values}),
            epochs=120,
            batch_size=256,
            verbose=0,
            callbacks=callbacks,
        )
        model.save(model_dir / KERAS_MODEL_PATH.name)
        p_cls, p_qty = model.predict(Xv_dense, verbose=0)
        keras_metrics = {
            "keras_val_mae_after_raw": float(mean_absolute_error(y_reg_val, p_qty.reshape(-1))),
            "keras_val_positive_rate": float(np.mean(p_cls.reshape(-1) >= 0.5)),
        }
        keras_trained = True

    # Always train a CPU-light fallback. The Streamlit app can use this if TensorFlow is unavailable.
    clf = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=0.0005, l1_ratio=0.08, max_iter=2500, tol=1e-3, class_weight="balanced", random_state=42)
    reg = SGDRegressor(loss="huber", penalty="elasticnet", alpha=0.0008, l1_ratio=0.05, max_iter=2500, tol=1e-3, random_state=42)
    clf.fit(Xt, y_cls_train)
    positive = y_reg_train > 0
    # Train quantity on all rows, but log target stabilizes outliers.
    reg.fit(Xt, np.log1p(y_reg_train))
    cls_prob = clf.predict_proba(Xv)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Xv)
    reg_raw = np.expm1(reg.predict(Xv)).clip(min=0)
    joblib.dump({"classifier": clf, "regressor": reg}, model_dir / SKLEARN_MODEL_PATH.name)

    pred_cls = (cls_prob >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_cls_val, pred_cls, average="binary", zero_division=0)
    meta = {
        "rows_trained": int(len(X)),
        "features": int(Xt.shape[1]),
        "positive_alloc_rate": float(y_cls.mean()),
        "fallback_val_mae_raw": float(mean_absolute_error(y_reg_val, reg_raw)),
        "fallback_precision": float(pr),
        "fallback_recall": float(rc),
        "fallback_f1": float(f1),
        "keras_trained": bool(keras_trained),
        **keras_metrics,
    }
    with open(model_dir / META_PATH.name, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


class AllocationPredictor:
    def __init__(self, model_dir: str | Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.pre = None
        self.keras_model = None
        self.fallback = None
        self.backend = "heuristic"
        self._load()

    def _load(self):
        if (self.model_dir / PREPROCESSOR_PATH.name).exists():
            self.pre = joblib.load(self.model_dir / PREPROCESSOR_PATH.name)
        tf = _try_tensorflow()
        if tf is not None and (self.model_dir / KERAS_MODEL_PATH.name).exists():
            self.keras_model = tf.keras.models.load_model(self.model_dir / KERAS_MODEL_PATH.name, compile=False)
            self.backend = "keras"
        elif (self.model_dir / SKLEARN_MODEL_PATH.name).exists():
            self.fallback = joblib.load(self.model_dir / SKLEARN_MODEL_PATH.name)
            self.backend = "sklearn_fallback"

    def predict(self, df: pd.DataFrame, schema_path: str = "feature_schema.json") -> tuple[np.ndarray | None, np.ndarray | None, str]:
        if self.pre is None or self.backend == "heuristic":
            return None, None, "heuristic"
        X, _, _, _ = build_model_frame(df, schema_path=schema_path, training=False)
        Xp = self.pre.transform(X.drop(columns=["__excel_row"], errors="ignore"))
        if self.backend == "keras" and self.keras_model is not None:
            dense = Xp.toarray() if hasattr(Xp, "toarray") else np.asarray(Xp)
            prob, raw = self.keras_model.predict(dense, verbose=0)
            return raw.reshape(-1), prob.reshape(-1), "keras"
        if self.backend == "sklearn_fallback" and self.fallback is not None:
            clf = self.fallback["classifier"]
            reg = self.fallback["regressor"]
            prob = clf.predict_proba(Xp)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Xp)
            raw = np.expm1(reg.predict(Xp)).clip(min=0)
            return raw.reshape(-1), prob.reshape(-1), "sklearn_fallback"
        return None, None, "heuristic"


def read_metadata(model_dir: str | Path = MODEL_DIR) -> Dict[str, Any]:
    path = Path(model_dir) / META_PATH.name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
