from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def norm_name(value: Any) -> str:
    """Normalize Excel headers so minor punctuation/spacing differences do not break mapping."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    index: int          # 1-based Excel index
    excel: str
    header: str


def load_schema(path: str | Path = "feature_schema.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def schema_columns(schema: Dict[str, Any]) -> Dict[str, ColumnSpec]:
    out: Dict[str, ColumnSpec] = {}
    for key, raw in schema["columns"].items():
        out[key] = ColumnSpec(key=key, index=int(raw["index"]), excel=raw["excel"], header=raw["header"])
    return out


def build_header_map(headers: List[Any], schema: Dict[str, Any]) -> Dict[str, int]:
    """Return semantic key -> zero-based dataframe column index.

    The original workbooks contain hidden/helper columns. Depending on how a .xlsb
    is read, a hidden/blank column can be present, omitted, or renamed. To avoid
    off-by-one errors, this resolver now prefers normalized header-name matching
    and uses hard-coded Excel positions only as a fallback. That keeps the app
    stable when hidden columns exist or when a new file contains the same headers
    at slightly shifted positions.
    """
    cols = schema_columns(schema)
    normalized = [norm_name(h) for h in headers]
    lookup: Dict[str, List[int]] = {}
    for i, h in enumerate(normalized):
        lookup.setdefault(h, []).append(i)

    aliases: Dict[str, List[str]] = {
        "item": ["item", "item_id", "item_number", "sku", "product_id"],
        "site": ["site", "site_id", "store", "store_id", "location"],
        "dc_avail": ["dc_avail", "dc_available", "dc_available_units", "available_dc", "dc_oh"],
        "flm": ["flm", "final_layer_multiple", "alloc_multiple", "allocation_multiple"],
        "alloc_rec": ["alloc_rec", "alloc_recommendation", "allocation_rec", "recommended_alloc"],
        "final_alloc": ["final_alloc", "final_alloc_", "final_allocation"],
        "left_dc": ["left_dc", "left_in_dc", "dc_left", "remaining_dc"],
        "final_supply": ["final_supply", "final_supply_units"],
        "demand_check": ["demand_check", "demand_review", "demand_flag"],
        "helper": ["helper", "helper_check", "allocation_helper"],
        "proj_demand": ["proj_demand", "projected_demand", "projection_demand"],
    }

    resolved: Dict[str, int] = {}
    for key, spec in cols.items():
        candidate_names = [norm_name(spec.header)] + aliases.get(key, [])
        # Header-name match first. This is safest when hidden columns are present.
        for target in candidate_names:
            if target in lookup:
                # Duplicate names are usually generic columns; use the last one to
                # match the original allocation template behavior.
                resolved[key] = lookup[target][-1]
                break
        if key in resolved:
            continue

        # Position fallback for workbooks with blank/renamed headers.
        preferred = spec.index - 1
        if 0 <= preferred < len(headers):
            resolved[key] = preferred
    return resolved


def safe_col(df, col_name: str, default=None):
    if col_name in df.columns:
        return df[col_name]
    return default
