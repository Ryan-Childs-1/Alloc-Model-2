from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from schema import build_header_map, load_schema, norm_name


def _to_num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _text(s):
    return s.fillna("").astype(str)


def canonicalize_dataframe(df: pd.DataFrame, schema_path: str = "feature_schema.json") -> tuple[pd.DataFrame, Dict[str, str]]:
    """
    Adds canonical semantic columns prefixed with c__ while retaining all original columns.
    Returns dataframe and semantic key -> original column name mapping.
    """
    schema = load_schema(schema_path)
    headers = list(df.columns)
    # __excel_row is inserted before the real headers; adjust mapping by using headers after it.
    real_headers = [h for h in headers if h != "__excel_row"]
    resolved = build_header_map(real_headers, schema)
    mapping: Dict[str, str] = {}
    out = df.copy()
    for key, zero_idx in resolved.items():
        if zero_idx < len(real_headers):
            src = real_headers[zero_idx]
            mapping[key] = src
            out[f"c__{key}"] = out[src]
    return out, mapping


def add_formula_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def num(key, default=0.0):
        col = f"c__{key}"
        if col in out:
            return pd.to_numeric(out[col], errors="coerce").fillna(default).astype(float)
        return pd.Series(default, index=out.index, dtype=float)

    def txt(key):
        col = f"c__{key}"
        if col in out:
            return out[col].fillna("").astype(str)
        return pd.Series("", index=out.index, dtype=str)

    out["f__flm"] = num("flm", 1.0).replace(0, np.nan).fillna(1.0).clip(lower=1.0)
    out["f__l30"] = num("l30")
    out["f__d30"] = num("d30")
    out["f__d60"] = num("d60")
    out["f__lw"] = num("lw")
    out["f__ttm"] = num("ttm")
    out["f__supply"] = num("supply")
    out["f__qoh"] = num("qoh")
    out["f__dc_avail"] = num("dc_avail")
    out["f__proj_demand"] = num("proj_demand")
    out["f__alloc_rec"] = num("alloc_rec")
    out["f__left_dc_cached"] = num("left_dc")
    out["f__final_supply_cached"] = num("final_supply")
    out["f__rank"] = num("rank")
    out["f__days"] = num("days", 30.0)

    # Demand basis: demand-protective but not blind to D60=0 cases.
    out["f__demand_basis"] = np.maximum.reduce([
        out["f__d60"].values,
        out["f__proj_demand"].values,
        (out["f__l30"] * 2.0).values,
        (out["f__ttm"] / 6.0).values,
        (out["f__lw"] * 8.0).values,
    ])
    out["f__need_gap"] = (out["f__demand_basis"] + out["f__flm"] - out["f__supply"]).clip(lower=0)
    out["f__supply_to_demand"] = out["f__supply"] / out["f__demand_basis"].replace(0, np.nan)
    out["f__supply_to_demand"] = out["f__supply_to_demand"].replace([np.inf, -np.inf], np.nan).fillna(99)
    out["f__alloc_rec_flm_units"] = out["f__alloc_rec"] / out["f__flm"].replace(0, 1)
    out["f__need_flm_units"] = out["f__need_gap"] / out["f__flm"].replace(0, 1)
    out["f__dc_flm_units"] = out["f__dc_avail"] / out["f__flm"].replace(0, 1)

    flag = txt("flag").str.upper()
    demand_check = txt("demand_check").str.upper()
    helper = txt("helper").str.upper()
    out["f__is_allocate"] = flag.str.contains("ALLOCATE", regex=False) & ~flag.str.contains("NO ALLOC", regex=False)
    out["f__is_review"] = flag.str.contains("REVIEW", regex=False)
    out["f__is_no_alloc"] = flag.str.contains("NO ALLOC", regex=False) | flag.str.contains("Z -", regex=False)
    out["f__eligible"] = (out["f__is_allocate"] | out["f__is_review"]) & ~out["f__is_no_alloc"]
    out["f__has_demand_warning"] = demand_check.str.len().gt(0) | helper.str.contains("TOO HIGH|REVIEW|DEMAND|SINGLE FLM", regex=True)
    out["f__single_flm_warning"] = helper.str.contains("SINGLE FLM|TOO HIGH", regex=True)

    # Group features by item and optional product/site context.
    item_col = "c__item" if "c__item" in out else None
    if item_col:
        item = out[item_col]
        out["g__rows_per_item"] = out.groupby(item_col)[item_col].transform("size")
        out["g__item_total_demand"] = out.groupby(item_col)["f__demand_basis"].transform("sum")
        out["g__item_total_supply"] = out.groupby(item_col)["f__supply"].transform("sum")
        out["g__item_max_dc_avail"] = out.groupby(item_col)["f__dc_avail"].transform("max")
        out["g__item_need_rank"] = out.groupby(item_col)["f__need_gap"].rank(method="dense", ascending=False)
        out["g__item_demand_share"] = out["f__demand_basis"] / out["g__item_total_demand"].replace(0, np.nan)
        out["g__item_demand_share"] = out["g__item_demand_share"].fillna(0)
    else:
        out["g__rows_per_item"] = 1
        out["g__item_total_demand"] = out["f__demand_basis"]
        out["g__item_total_supply"] = out["f__supply"]
        out["g__item_max_dc_avail"] = out["f__dc_avail"]
        out["g__item_need_rank"] = 1
        out["g__item_demand_share"] = 1

    return out


def build_model_frame(df: pd.DataFrame, schema_path: str = "feature_schema.json", training: bool = True) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, Dict[str, str]]:
    canon, mapping = canonicalize_dataframe(df, schema_path)
    feat = add_formula_features(canon)

    # Preferred categorical fields available in the actual template.
    categorical_candidates = [
        "Vendor", "Vendor Site Id", "Brand", "Dcl", "Class Name", "Line Name", "Product ID",
        "Status", "Status 300", "Site Name", "State", "Region", "Zone", "Buyer Name",
        "Planner Code", "Private Label", "Season Code", "Store Size", "New", "Store Flag",
        "SKU Flag", "c__flag", "c__demand_check", "c__helper"
    ]
    categorical_cols = [c for c in categorical_candidates if c in feat.columns]

    leakage_cols = {"f__left_dc_cached", "f__final_supply_cached"}
    numeric_cols = [c for c in feat.columns if (c.startswith("f__") or c.startswith("g__")) and c not in leakage_cols]
    # Include known numeric original fields not in canonical features. Avoid target/leakage columns derived from Final Alloc.
    for c in feat.columns:
        if c.startswith("__") or c.startswith("c__"):
            continue
        if norm_name(c) in {"final_alloc", "final_alloc_", "final_allocation", "left_dc", "final_supply", "final_all_units"}:
            continue
        if c in categorical_cols:
            continue
        if c not in numeric_cols:
            vals = pd.to_numeric(feat[c], errors="coerce")
            if vals.notna().mean() > 0.80:
                feat[f"n__{c}"] = vals.fillna(0)
                numeric_cols.append(f"n__{c}")

    X = pd.concat([
        feat[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0),
        feat[categorical_cols].fillna("").astype(str).add_prefix("cat__")
    ], axis=1)
    X["__excel_row"] = feat["__excel_row"].values if "__excel_row" in feat else np.arange(len(feat)) + 3

    y_reg = None
    y_cls = None
    if training and "c__final_alloc" in feat.columns:
        y_raw = pd.to_numeric(feat["c__final_alloc"], errors="coerce").fillna(0).clip(lower=0)
        y_reg = y_raw
        y_cls = (y_raw > 0).astype(int)
    return X, y_reg, y_cls, mapping
