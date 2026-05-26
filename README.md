# Allocation AI v2

Allocation AI v2 is a Windows/Excel-based Streamlit app for Daily Allocation `.xlsb` workbooks. It preserves the original workbook structure, formulas, formatting, and binary `.xlsb` format while overwriting only the `Final Alloc.` column with AI-generated allocation values.

## What this version does

- Keeps the output file as `.xlsb`
- Writes only `Final Alloc.`
- Returns blank cells for no-allocation rows
- Preserves workbook formulas and formatting through Microsoft Excel COM automation
- Uses a hybrid AI system:
  - Keras neural classifier/regressor when TensorFlow is installed
  - CPU fallback model using scikit-learn
  - Rule-backed allocation engine if no model is trained yet
- Simulates `Left DC` sequentially by item so inventory decreases as Final Alloc values are entered
- Uses demand-protective caps so Final Supply does not exceed demand by more than the configured FLM cushion
- Treats blank FLM as `1`
- Treats rows without `Allocate` or `Review` as no-allocation rows
- Optional audit sheet explains every allocation decision

## Workbook assumptions

Main sheet: `3.3 Working Table`

Important mapped columns:

| Field | Column | Header |
|---|---:|---|
| Item | O | Item |
| L30 | AK | L30 |
| D60 | AM | D60 |
| TTM | AO | TTM |
| Supply | AQ | Supply |
| DC Avail | AX | Dc Avail |
| FLM | BK | FLM |
| Proj. Demand | BM | Proj. Demand |
| Alloc. Rec. | BN | Alloc. Rec. |
| Flag | BR | Flag |
| Final Alloc. | BS | Final Alloc. |
| Left DC | BT | Left DC |
| Final Supply | BU | Final Supply |
| Demand Check | BZ | Demand Check |
| Helper | CA | Helper |

## Install

Run this on a Windows computer with Microsoft Excel installed.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow is included in `requirements.txt`. If TensorFlow installation is not available on your machine, the app still trains and runs the scikit-learn fallback model.

## Run

Double-click:

```bat
run_app.bat
```

Or run:

```bat
streamlit run app.py
```

## Recommended workflow

1. Open the app.
2. Go to **Train / Retrain Model**.
3. Upload historical completed `.xlsb` workbooks where existing `Final Alloc.` values are the correct ground truth.
4. Train the model.
5. Go to **Predict Allocation**.
6. Upload a new `.xlsb` workbook.
7. Choose whether to include the optional audit sheet.
8. Run Allocation AI.
9. Download the completed `.xlsb` workbook.

## Production notes

- Close the workbook in Excel before running prediction.
- Use **Excel COM / xlwings** read mode for production.
- Use **pyxlsb** read mode only for fast preview/training from cached formula values.
- The app intentionally writes blanks, not zeros, when no allocation is recommended.
- The default rounding mode is `floor` because it is safer and more demand-protective.

## Files

- `app.py` — Streamlit interface
- `excel_bridge.py` — `.xlsb` read/write bridge using Excel COM and pyxlsb fallback
- `feature_engineering.py` — formula-aware feature construction and group features
- `allocation_engine.py` — sequential Left DC simulation and allocation constraints
- `model_utils.py` — Keras neural network training/loading plus sklearn fallback
- `train_model.py` — command-line model training
- `schema.py` — column mapping helpers
- `feature_schema.json` — locked workbook schema
- `requirements.txt` — dependencies
- `run_app.bat` — Windows launcher

## Command-line training option

```bat
python train_model.py "Daily Allocation - 127, 128, 706, 134 - 5.26.2026.xlsb" "Daily Allocation - 130, 135, 114, 115, 153 - 5.26.2026.xlsb" "Daily Allocation - 132, 150 - 5.26.2026.xlsb"
```

The app stores trained artifacts in a local `models` folder.
