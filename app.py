from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from allocation_engine import AllocationConfig, simulate_allocations
from excel_bridge import ExcelComBridge, read_xlsb_values_pyxlsb, save_uploaded_file
from model_utils import AllocationPredictor, read_metadata, train_models_from_frames
from schema import load_schema

st.set_page_config(page_title="Allocation AI v2", page_icon="🧠", layout="wide")

SCHEMA_PATH = "feature_schema.json"
MODEL_DIR = Path("models")

st.title("🧠 Allocation AI v2")
st.caption("Hybrid neural allocation engine for Sportsman's-style Daily Allocation .xlsb workbooks")

with st.sidebar:
    st.header("Settings")
    include_audit = st.checkbox("Include optional audit sheet", value=False)
    visible_excel = st.checkbox("Show Excel while writing file", value=False)
    st.divider()
    st.subheader("Allocation behavior")
    min_prob = st.slider("Minimum allocation probability", 0.10, 0.90, 0.42, 0.01)
    cap_extra = st.slider("Demand cap extra FLM", 0.0, 3.0, 1.0, 0.25)
    round_mode = st.selectbox("FLM rounding mode", ["floor", "nearest"], index=0)
    st.caption("Floor is safest and most demand-protective. Nearest can be slightly more aggressive.")

schema = load_schema(SCHEMA_PATH)
metadata = read_metadata(MODEL_DIR)
if metadata:
    st.success(f"Loaded model metadata: {metadata.get('rows_trained', 'unknown')} rows trained · backend will auto-detect.")
else:
    st.warning("No trained model found yet. Prediction will still work using the rule-backed allocation engine, but training is recommended.")

tab_predict, tab_train, tab_schema = st.tabs(["Predict Allocation", "Train / Retrain Model", "Workbook Schema"])

with tab_predict:
    st.subheader("1. Upload a .xlsb workbook")
    uploaded = st.file_uploader("Upload Daily Allocation workbook", type=["xlsb"], key="predict_file")
    if uploaded is not None:
        input_path = save_uploaded_file(uploaded, suffix=".xlsb")
        st.write(f"Uploaded: **{uploaded.name}**")
        read_mode = st.radio(
            "Read method",
            ["Excel COM / xlwings", "Fast cached values / pyxlsb"],
            index=0,
            help="Use Excel COM on your Windows work computer for most accurate values and formula state. pyxlsb is faster but read-only and uses cached formula values.",
        )
        if st.button("Run Allocation AI", type="primary"):
            try:
                with st.spinner("Reading workbook..."):
                    if read_mode.startswith("Excel"):
                        bridge = ExcelComBridge(visible=visible_excel)
                        df = bridge.read_table(input_path, schema["main_sheet"], schema["header_row"], schema["first_data_row"])
                    else:
                        df = read_xlsb_values_pyxlsb(input_path, schema["main_sheet"], schema["header_row"], schema["first_data_row"])

                with st.spinner("Predicting allocation need and applying workbook-aware constraints..."):
                    predictor = AllocationPredictor(MODEL_DIR)
                    raw, prob, backend = predictor.predict(df, schema_path=SCHEMA_PATH)
                    cfg = AllocationConfig(min_probability=min_prob, demand_cap_extra_flm=cap_extra, round_mode=round_mode)
                    pred_df, audit_df = simulate_allocations(df, raw, prob, schema_path=SCHEMA_PATH, config=cfg)

                st.info(f"Prediction backend used: **{backend}**")
                c1, c2, c3, c4 = st.columns(4)
                allocated_rows = pred_df["ai_final_alloc"].fillna(0).gt(0).sum()
                total_units = pred_df["ai_final_alloc"].fillna(0).sum()
                c1.metric("Rows processed", f"{len(pred_df):,}")
                c2.metric("Rows allocated", f"{allocated_rows:,}")
                c3.metric("AI final units", f"{total_units:,.0f}")
                c4.metric("Blank rows", f"{len(pred_df)-allocated_rows:,}")

                st.subheader("Audit preview")
                st.dataframe(audit_df.head(250), use_container_width=True)

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
                    "**Common fix:** Make sure this app is running on Windows with Microsoft Excel installed, and close the workbook if it is already open."
                )

with tab_train:
    st.subheader("Train from historical allocation workbooks")
    st.write("Upload one or more completed historical `.xlsb` workbooks where existing Final Alloc values are the ground truth.")
    training_uploads = st.file_uploader("Historical training workbooks", type=["xlsb"], accept_multiple_files=True, key="train_files")
    max_rows = st.number_input("Optional max rows per file for quick testing; 0 means all rows", min_value=0, value=0, step=500)
    if training_uploads and st.button("Train / Retrain Allocation Model", type="primary"):
        try:
            frames = []
            progress = st.progress(0)
            for i, up in enumerate(training_uploads):
                path = save_uploaded_file(up, suffix=".xlsb")
                st.write(f"Reading **{up.name}**...")
                frames.append(read_xlsb_values_pyxlsb(path, schema["main_sheet"], schema["header_row"], schema["first_data_row"], max_rows=None if max_rows == 0 else int(max_rows)))
                progress.progress((i + 1) / len(training_uploads))
            with st.spinner("Training Keras neural model when TensorFlow is installed, plus CPU fallback model..."):
                meta = train_models_from_frames(frames, schema_path=SCHEMA_PATH, model_dir=MODEL_DIR)
            st.success("Training complete.")
            st.json(meta)
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
        "The app preserves `.xlsb` by using real Excel through xlwings/COM during write-back. The model reads formula-derived values, simulates Left DC by item, and writes only the Final Alloc column."
    )
