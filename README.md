# Parkinson Spiral Research Prototype

This project is designed as a stronger research/PhD demonstration than a simple image classifier.

## Research design

The pipeline includes:

- deterministic preprocessing
- handcrafted spiral geometry features
- SVM, Random Forest and XGBoost
- subject-level hold-out evaluation
- sensitivity and specificity
- ROC-AUC and PR-AUC
- Brier score
- bootstrap 95% confidence interval for held-out ROC-AUC
- held-out prediction export
- indeterminate zone around the decision boundary
- Streamlit training and inference UI

## Dataset

Preferred:

```text
dataset.zip
├── control/
│   ├── subject01_spiral.png
│   ├── subject02_spiral.png
│   └── ...
└── parkinson/
    ├── subject101_spiral.png
    ├── subject102_spiral.png
    └── ...
```

If a participant has multiple drawings, use a consistent subject identifier in filenames and adapt `infer_subject()` in `src/research_train.py` to your exact naming convention.

## Run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Research warning

A PhD-level presentation should not claim clinical diagnostic performance from a small public or convenience dataset. The project should report subject-level separation, external validation where possible, confidence intervals, class balance, missing/skipped images, and limitations.

The score displayed by the application is an ML model score. It is not a medical probability and is not a diagnosis.
