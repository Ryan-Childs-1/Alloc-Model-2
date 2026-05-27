from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from allocation_engine import AllocationConfig, simulate_allocations
from excel_bridge import ExcelComBridge, read_xlsb_values_pyxlsb, save_uploaded_file
from model_utils import AllocationPredictor, read_metadata, train_models_from_frames
from schema import load_schema

st.set_page_config(page_title="Allocation AI v2 Keras", page_icon="🧠", layout="wide")

SCHEMA_PATH = "feature_schema.json"
MODEL_DIR = Path(".")

st.title("🧠 Allocation AI v2 — Keras Neural Network Build")
st.caption("Hybrid Keras neural allocation engine for Daily Allocation .xlsb workbooks. No JAX. TensorFlow not required when using the PyTorch backend.")

with st.sidebar:
    st.header("Settings")
    include_audit = st.checkbox("Include optional audit sheet", value=False)
    visible_excel = st.checkbox("Show Excel while writing file", value=False)
    st.divider()
    st.subheader("Allocation behavior")
    min_prob = st.slider("Minimum allocation probability", 0.10, 0.90, 0.42, 0.01)
    cap_extra = st.slider("Demand cap extra FLM", 0.0, 3.0, 1.0, 0.25)
    round_mode = st.selectbox("FLM rounding mode", ["floor", "nearest"], index=0)
    st.caption("Floor is safer and more demand-protective. Nearest can be slightly more aggressive.")

schema = load_schema(SCHEMA_PATH)
metadata = read_metadata(MODEL_DIR)

if metadata:
    st.success(
        f"Loaded model metadata: {metadata.get('rows_trained', 'unknown')} rows trained · "
        f"primary backend: {metadata.get('primary_backend', 'auto')}"
    )
else:
    st.warning("No trained model artifacts found yet. The app can still run with the rule-backed allocation engine, but training is recommended.")

if platform.system() != "Windows":
    st.info(
        "This app can launch, read cached .xlsb values, train, and preview audits on this platform. True preserved .xlsb write-back requires Windows with Microsoft Excel installed."
    )

tab_predict, tab_train, tab_schema, tab_help = st.tabs([
    "Predict Allocation",
    "Train / Retrain Model",
    "Workbook Schema",
    "Install Help",
])

with tab_predict:
    st.subheader("1. Upload a .xlsb workbook")
    uploaded = st.file_uploader("Upload Daily Allocation workbook", type=["xlsb"], key="predict_file")
    if uploaded is not None:
        input_path = save_uploaded_file(uploaded, suffix=".xlsb")
        st.write(f"Uploaded: **{uploaded.name}**")
        read_mode = st.radio(
            "Read method",
            ["Excel COM / xlwings", "Fast cached values / pyxlsb"],
            index=0 if platform.system() == "Windows" else 1,
            help="Use Excel COM on your Windows work computer for production. pyxlsb is faster but read-only and uses cached formula values.",
        )
        if st.button("Run Allocation AI", type="primary"):
            try:
                with st.spinner("Reading workbook..."):
                    if read_mode.startswith("Excel"):
                        bridge = ExcelComBridge(visible=visible_excel)
                        df = bridge.read_table(input_path, schema["main_sheet"], schema["header_row"], schema["first_data_row"])
                    else:
                        df = read_xlsb_values_pyxlsb(input_path, schema["main_sheet"], schema["header_row"], schema["first_data_row"])

                with st.spinner("Running Keras/ensemble prediction and workbook-aware allocation constraints..."):
                    predictor = AllocationPredictor(MODEL_DIR)
                    raw, prob, backend = predictor.predict(df, schema_path=SCHEMA_PATH)
                    cfg = AllocationConfig(min_probability=min_prob, demand_cap_extra_flm=cap_extra, round_mode=round_mode)
                    pred_df, audit_df = simulate_allocations(df, raw, prob, schema_path=SCHEMA_PATH, config=cfg)

                st.info(f"Prediction backend used: **{backend}**")
                c1, c2, c3, c4 = st.columns(4)
                allocated_rows = int(pred_df["ai_final_alloc"].fillna(0).gt(0).sum())
                total_units = float(pred_df["ai_final_alloc"].fillna(0).sum())
                c1.metric("Rows processed", f"{len(pred_df):,}")
                c2.metric("Rows allocated", f"{allocated_rows:,}")
                c3.metric("AI final units", f"{total_units:,.0f}")
                c4.metric("Blank rows", f"{len(pred_df)-allocated_rows:,}")

                st.subheader("Audit preview")
                st.dataframe(audit_df.head(300), use_container_width=True)

                if platform.system() != "Windows":
                    st.warning("Downloadable preserved .xlsb write-back is disabled here because Excel COM requires Windows + Microsoft Excel.")
                    st.download_button(
                        "Download audit CSV instead",
                        data=audit_df.to_csv(index=False).encode("utf-8"),
                        file_name=uploaded.name.replace(".xlsb", " - Allocation AI Audit.csv"),
                        mime="text/csv",
                    )
                else:
                    with st.spinner("Writing only Final Alloc back to .xlsb and preserving workbook structure..."):
                        out_name = uploaded.name.replace(".xlsb", " - Allocation AI Output.xlsb")
                        out_path = Path(tempfile.gettempdir()) / out_name
                        bridge = ExcelComBridge(visible=visible_excel)
                        final_path = bridge.write_final_alloc_xlsb(
                            input_path,
                            out_path,
                            pred_df,
                            schema_path=SCHEMA_PATH,
                            include_audit=include_audit,
                            audit_df=audit_df,
                            sheet_name=schema["main_sheet"],
                        )
                    with open(final_path, "rb") as f:
                        st.download_button(
                            "Download completed .xlsb workbook",
                            data=f.read(),
                            file_name=out_name,
                            mime="application/vnd.ms-excel.sheet.binary.macroEnabled.12",
                        )
                    st.success("Completed workbook created. Only Final Alloc was overwritten; no-allocation rows were left blank.")
            except Exception as exc:
                st.error("Allocation run failed.")
                st.exception(exc)
                st.markdown(
                    "**Common fixes:** Close the workbook if it is already open, run on Windows with Excel installed for .xlsb write-back, or use the pyxlsb read mode for training/preview."
                )

with tab_train:
    st.subheader("Train from historical allocation workbooks")
    st.write("Upload one or more completed historical `.xlsb` workbooks where existing Final Alloc values are the ground truth.")
    training_uploads = st.file_uploader("Historical training workbooks", type=["xlsb"], accept_multiple_files=True, key="train_files")
    max_rows = st.number_input("Optional max rows per file for quick testing; 0 means all rows", min_value=0, value=0, step=500)
    if training_uploads and st.button("Train / Retrain Keras Allocation Model", type="primary"):
        try:
            frames = []
            progress = st.progress(0)
            for i, up in enumerate(training_uploads):
                path = save_uploaded_file(up, suffix=".xlsb")
                st.write(f"Reading **{up.name}**...")
                frames.append(
                    read_xlsb_values_pyxlsb(
                        path,
                        schema["main_sheet"],
                        schema["header_row"],
                        schema["first_data_row"],
                        max_rows=None if max_rows == 0 else int(max_rows),
                    )
                )
                progress.progress((i + 1) / len(training_uploads))
            with st.spinner("Training Keras neural network plus ensemble/MLP fallback..."):
                meta = train_models_from_frames(frames, schema_path=SCHEMA_PATH, model_dir=MODEL_DIR)
            st.success("Training complete.")
            st.json(meta)
            st.info("Model artifacts were saved flat in the app folder: preprocessor.joblib, allocation_model_keras.keras, allocation_fallback_ensemble.joblib, model_metadata.json.")
        except Exception as exc:
            st.error("Training failed.")
            st.exception(exc)

with tab_schema:
    st.subheader("Mapped workbook columns")
    cols = []
    for key, spec in schema["columns"].items():
        cols.append({"Semantic Field": key, "Excel Column": spec["excel"], "Index": spec["index"], "Expected Header": spec["header"]})
    st.dataframe(pd.DataFrame(cols), use_container_width=True)
    st.markdown(
        "The app preserves `.xlsb` through real Excel on Windows during write-back. The model reads formula-derived values, simulates Left DC by Item, and writes only Final Alloc."
    )

with tab_help:
    st.subheader("Install and launch")
    st.markdown(
        """
### Streamlit-compatible install

```bat
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows production write-back

`requirements.txt` uses Windows-only environment markers for `xlwings` and `pywin32`, so they are skipped on Streamlit Cloud/Linux but installed on Windows.

### If Keras or Torch is blocked by your machine

Use the fallback-only requirements. This still uses an advanced sklearn ensemble plus MLP and the same allocation simulator:

```bat
pip install -r requirements_fallback_only.txt
streamlit run app.py
```
        """
    )
