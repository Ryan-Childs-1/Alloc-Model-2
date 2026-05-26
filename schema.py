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
    """
    Return semantic key -> zero-based dataframe column index.
    Uses preferred hard-coded positions first, then normalized header fallback.
    """
    cols = schema_columns(schema)
    normalized = [norm_name(h) for h in headers]
    lookup: Dict[str, List[int]] = {}
    for i, h in enumerate(normalized):
        lookup.setdefault(h, []).append(i)

    resolved: Dict[str, int] = {}
    for key, spec in cols.items():
        preferred = spec.index - 1
        if 0 <= preferred < len(headers):
            # Trust position if the cell has either expected header or a duplicate generic header.
            resolved[key] = preferred
            continue
        target = norm_name(spec.header)
        if target in lookup:
            # For duplicate names like FLM/MIL, use the last occurrence by default.
            resolved[key] = lookup[target][-1]
    return resolved


def safe_col(df, col_name: str, default=None):
    if col_name in df.columns:
        return df[col_name]
    return default
