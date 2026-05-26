from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from schema import build_header_map, load_schema


def _clean_header(h: Any, idx: int) -> str:
    if h is None or str(h).strip() == "":
        return f"__blank_{idx+1}"
    text = str(h).strip()
    # Make duplicate headers unique while keeping human-readable names.
    return text


def _dedupe_headers(headers: List[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, h in enumerate(headers):
        base = _clean_header(h, i)
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base}__{n+1}")
    return out


def read_xlsb_values_pyxlsb(path: str | Path, sheet_name: str, header_row: int = 2, first_data_row: int = 3, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Fast read-only fallback for .xlsb cached values. Does not preserve formulas/styles."""
    from pyxlsb import open_workbook

    rows: List[List[Any]] = []
    headers: Optional[List[Any]] = None
    with open_workbook(str(path)) as wb:
        with wb.get_sheet(sheet_name) as sh:
            blank_streak = 0
            for ridx, row in enumerate(sh.rows(), start=1):
                vals = [c.v for c in row]
                if ridx == header_row:
                    headers = vals
                    continue
                if ridx < first_data_row:
                    continue
                if headers is None:
                    continue
                if len(vals) < len(headers):
                    vals += [None] * (len(headers) - len(vals))
                vals = vals[: len(headers)]
                if all(v is None or str(v).strip() == "" for v in vals):
                    blank_streak += 1
                    if blank_streak >= 250 and rows:
                        break
                    continue
                blank_streak = 0
                rows.append(vals)
                if max_rows and len(rows) >= max_rows:
                    break
    if headers is None:
        raise ValueError(f"Could not find header row {header_row} in {sheet_name}.")
    df = pd.DataFrame(rows, columns=_dedupe_headers(headers))
    df.insert(0, "__excel_row", range(first_data_row, first_data_row + len(df)))
    return df


class ExcelComBridge:
    """
    Windows-only .xlsb bridge using real Microsoft Excel through xlwings/COM.
    This is the production path because it preserves xlsb format, formulas, styles, and sheets.
    """

    def __init__(self, visible: bool = False):
        self.visible = visible

    def _xlwings(self):
        try:
            import xlwings as xw
            return xw
        except Exception as exc:
            raise RuntimeError("xlwings is required for .xlsb write-back. Install requirements.txt on Windows with Excel installed.") from exc

    def read_table(self, workbook_path: str | Path, sheet_name: str, header_row: int = 2, first_data_row: int = 3) -> pd.DataFrame:
        xw = self._xlwings()
        app = xw.App(visible=self.visible, add_book=False)
        app.display_alerts = False
        app.screen_updating = False
        try:
            wb = app.books.open(str(workbook_path), update_links=False, read_only=True)
            sht = wb.sheets[sheet_name]
            used = sht.used_range
            nrows = used.last_cell.row
            ncols = used.last_cell.column
            headers = sht.range((header_row, 1), (header_row, ncols)).value
            if not isinstance(headers, list):
                headers = [headers]
            values = sht.range((first_data_row, 1), (nrows, ncols)).value
            if values is None:
                values = []
            if values and not isinstance(values[0], list):
                values = [values]
            clean_headers = _dedupe_headers(headers)
            df = pd.DataFrame(values, columns=clean_headers)
            df.insert(0, "__excel_row", range(first_data_row, first_data_row + len(df)))
            # Remove fully empty trailing rows.
            body_cols = [c for c in df.columns if c != "__excel_row"]
            df = df.dropna(how="all", subset=body_cols).reset_index(drop=True)
            wb.close()
            return df
        finally:
            app.quit()

    def write_final_alloc_xlsb(
        self,
        input_path: str | Path,
        output_path: str | Path,
        predictions: pd.DataFrame,
        schema_path: str | Path = "feature_schema.json",
        include_audit: bool = False,
        audit_df: Optional[pd.DataFrame] = None,
        sheet_name: Optional[str] = None,
    ) -> str:
        xw = self._xlwings()
        schema = load_schema(schema_path)
        sheet_name = sheet_name or schema.get("main_sheet", "3.3 Working Table")
        final_col = int(schema["columns"]["final_alloc"]["index"])
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        app = xw.App(visible=self.visible, add_book=False)
        app.display_alerts = False
        app.screen_updating = False
        try:
            wb = app.books.open(str(input_path), update_links=False, read_only=False)
            sht = wb.sheets[sheet_name]
            # Write only Final Alloc cells. None clears to blank.
            for _, row in predictions.iterrows():
                excel_row = int(row["__excel_row"])
                val = row.get("ai_final_alloc", None)
                if pd.isna(val) or float(val) <= 0:
                    sht.range((excel_row, final_col)).value = None
                else:
                    # Keep integers clean in Excel.
                    fval = float(val)
                    sht.range((excel_row, final_col)).value = int(round(fval)) if abs(fval - round(fval)) < 1e-9 else fval

            if include_audit and audit_df is not None:
                name = "Allocation AI Audit"
                for s in list(wb.sheets):
                    if s.name == name:
                        s.delete()
                audit = wb.sheets.add(name, after=wb.sheets[-1])
                audit.range("A1").value = [audit_df.columns.tolist()] + audit_df.fillna("").values.tolist()
                try:
                    audit.autofit()
                    audit.range("A1").api.Font.Bold = True
                    audit.range("A1").api.AutoFilter()
                except Exception:
                    pass

            try:
                app.api.CalculateFullRebuild()
            except Exception:
                try:
                    wb.app.calculate()
                except Exception:
                    pass
            wb.save(output_path)
            wb.close()
            return output_path
        finally:
            app.quit()


def save_uploaded_file(uploaded_file, suffix: str = ".xlsb") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
