from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from feature_engineering import canonicalize_dataframe, add_formula_features


@dataclass
class AllocationConfig:
    min_probability: float = 0.42
    review_probability_boost: float = -0.03
    demand_cap_extra_flm: float = 1.0
    single_flm_warning_requires_probability: float = 0.58
    blank_zero_allocations: bool = True
    round_mode: str = "floor"  # floor is safest for demand-protective behavior.


def _round_to_flm(value: float, flm: float, mode: str = "floor") -> float:
    if value is None or not np.isfinite(value) or value <= 0:
        return 0.0
    flm = max(float(flm or 1), 1.0)
    units = value / flm
    if mode == "nearest":
        rounded_units = round(units)
    else:
        rounded_units = np.floor(units + 1e-9)
    if rounded_units <= 0 and value >= 0.65 * flm:
        rounded_units = 1
    return float(max(0, rounded_units) * flm)


def _raw_prediction_from_heuristic(feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rule-backed fallback if no trained model is available."""
    eligible = feat["f__eligible"].astype(bool).values
    raw = np.minimum(feat["f__alloc_rec"].values, feat["f__need_gap"].values)
    raw = np.where(raw <= 0, np.minimum(feat["f__alloc_rec"].values, feat["f__flm"].values), raw)
    # Probability proxy from need, flag, and alloc rec.
    prob = (
        0.20
        + 0.30 * feat["f__eligible"].astype(float).values
        + 0.20 * (feat["f__need_gap"].values > 0).astype(float)
        + 0.15 * (feat["f__alloc_rec"].values > 0).astype(float)
        + 0.10 * (feat["f__supply_to_demand"].values < 1.15).astype(float)
    )
    prob = np.where(eligible, prob, 0.02)
    return raw, np.clip(prob, 0, 0.98)


def simulate_allocations(
    original_df: pd.DataFrame,
    raw_alloc: Optional[np.ndarray] = None,
    alloc_probability: Optional[np.ndarray] = None,
    schema_path: str = "feature_schema.json",
    config: Optional[AllocationConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies sequential workbook-aware constraints. The model suggests need; this engine writes valid Final Alloc values.
    """
    config = config or AllocationConfig()
    canon, mapping = canonicalize_dataframe(original_df, schema_path)
    feat = add_formula_features(canon)
    n = len(feat)
    if raw_alloc is None or alloc_probability is None:
        raw_alloc, alloc_probability = _raw_prediction_from_heuristic(feat)
    raw_alloc = np.asarray(raw_alloc, dtype=float).reshape(-1)[:n]
    alloc_probability = np.asarray(alloc_probability, dtype=float).reshape(-1)[:n]

    # Initialize available inventory by item from DC Avail, falling back to cached Left DC.
    if "c__item" in feat.columns:
        item_key = feat["c__item"].fillna("__missing_item__").astype(str)
    else:
        item_key = pd.Series("__all__", index=feat.index)
    remaining_by_item: Dict[str, float] = {}
    for key, grp in feat.groupby(item_key):
        dc = pd.to_numeric(grp["f__dc_avail"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()
        if not np.isfinite(dc) or dc <= 0:
            dc = pd.to_numeric(grp["f__left_dc_cached"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()
        remaining_by_item[str(key)] = float(max(dc if np.isfinite(dc) else 0.0, 0.0))

    final_values: List[float] = []
    audit_rows: List[dict] = []

    for pos, row in feat.iterrows():
        excel_row = int(row.get("__excel_row", pos + 3))
        key = str(item_key.iloc[pos])
        remaining_before = float(remaining_by_item.get(key, 0.0))
        flm = float(max(row.get("f__flm", 1.0) or 1.0, 1.0))
        eligible = bool(row.get("f__eligible", False))
        is_review = bool(row.get("f__is_review", False))
        is_no_alloc = bool(row.get("f__is_no_alloc", False))
        p = float(alloc_probability[pos]) if pos < len(alloc_probability) else 0.0
        raw = float(raw_alloc[pos]) if pos < len(raw_alloc) else 0.0

        reasons = []
        if is_no_alloc or not eligible:
            final = 0.0
            reasons.append("Blanked: row is not Allocate/Review eligible")
        elif remaining_before <= 0:
            final = 0.0
            reasons.append("Blanked: no DC inventory remaining for item")
        else:
            threshold = config.min_probability + (config.review_probability_boost if is_review else 0.0)
            if p < threshold:
                final = 0.0
                reasons.append(f"Blanked: allocation probability {p:.2f} below threshold {threshold:.2f}")
            else:
                supply = float(row.get("f__supply", 0.0) or 0.0)
                demand_basis = float(row.get("f__demand_basis", 0.0) or 0.0)
                demand_cap = max(0.0, demand_basis + config.demand_cap_extra_flm * flm - supply)
                if bool(row.get("f__single_flm_warning", False)) and p < config.single_flm_warning_requires_probability:
                    demand_cap = min(demand_cap, max(0.0, 0.99 * flm))
                    reasons.append("Cautioned: helper flagged possible single-FLM over-allocation")
                capped = min(max(raw, 0.0), remaining_before, demand_cap)
                if capped < raw:
                    reasons.append("Capped by demand/remaining-DC guardrail")
                final = _round_to_flm(capped, flm, config.round_mode)
                if final > remaining_before:
                    final = _round_to_flm(remaining_before, flm, "floor")
                    reasons.append("Rounded down to remaining DC")
                if final <= 0:
                    reasons.append("Blanked: constrained allocation rounded to zero")
                else:
                    reasons.append("Approved")
        remaining_after = max(0.0, remaining_before - final)
        remaining_by_item[key] = remaining_after
        final_values.append(final)
        audit_rows.append({
            "Excel Row": excel_row,
            "Item": row.get("c__item", ""),
            "Site": row.get("c__site", ""),
            "Flag": row.get("c__flag", ""),
            "Original Final Alloc": row.get("c__final_alloc", ""),
            "AI Raw Allocation": raw,
            "AI Probability": p,
            "AI Final Alloc": final if final > 0 else "",
            "FLM": flm,
            "Demand Basis": row.get("f__demand_basis", 0.0),
            "Supply": row.get("f__supply", 0.0),
            "Left DC Before": remaining_before,
            "Left DC After": remaining_after,
            "Demand Check": row.get("c__demand_check", ""),
            "Helper": row.get("c__helper", ""),
            "Reason": "; ".join(dict.fromkeys(reasons)),
        })

    result = pd.DataFrame({
        "__excel_row": feat["__excel_row"].astype(int).values if "__excel_row" in feat else np.arange(n) + 3,
        "ai_final_alloc": [np.nan if v <= 0 and config.blank_zero_allocations else v for v in final_values],
    })
    audit = pd.DataFrame(audit_rows)
    return result, audit
