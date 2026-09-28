# Dataset format

Preferred ZIP:

```text
dataset.zip
├── control/
└── parkinson/
```

Use subject identifiers consistently. Example:

```text
control/P001_spiral.png
control/P002_spiral.png
parkinson/P101_spiral.png
parkinson/P102_spiral.png
```

If you have multiple drawings per participant:

```text
control/P001_trial1.png
control/P001_trial2.png
parkinson/P101_trial1.png
```

then modify `infer_subject()` so both P001 files map to subject P001. This prevents leakage between train and test.
