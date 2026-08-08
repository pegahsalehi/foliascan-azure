# Final Test Evaluation

FoliaScan evaluates the selected ResNet18 checkpoint on the held-out test split to produce final classification metrics and error-analysis outputs.

The checkpoint is selected using only training and validation data. The test split remains untouched until final evaluation so it can provide an independent estimate of model performance.

Test results should not be used repeatedly to tune transforms, hyperparameters, model architecture, class handling, or checkpoint selection. Those decisions belong to training and validation.

## Run Evaluation

Run the evaluator after the leakage-safe manifest, exported images, and selected checkpoint are available:

```powershell
poetry run python -m foliascan.evaluation.evaluate `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --checkpoint artifacts/training/resnet18_baseline_v1/best_model.pt `
  --output-dir artifacts/evaluation/resnet18_baseline_v1 `
  --device cuda
```

Use:

- `--device cpu` for CPU evaluation
- `--device cuda` to require CUDA
- `--device auto` to use CUDA when available and CPU otherwise

A non-empty output directory is rejected unless `--overwrite` is supplied.

## Evaluation Workflow

The evaluator restores the information required for deterministic inference from the selected checkpoint:

- model architecture
- model weights
- class mapping
- image size
- batch size

The optimizer state is not restored because evaluation does not update model parameters.

Only manifest rows with:

```text
split=test
```

are loaded.

The test DataLoader uses the same deterministic transform as validation:

- resize
- center crop
- tensor conversion
- ImageNet normalization

Inference runs with:

```python
model.eval()
torch.inference_mode()
```

## Metrics

The evaluator reports:

- overall accuracy
- per-class precision
- per-class recall
- per-class F1
- per-class support
- macro precision, recall, and F1
- weighted precision, recall, and F1
- raw-count confusion matrix

Accuracy measures the fraction of test samples classified correctly.

Precision measures how often predictions for a class are correct.

Recall measures how many samples of a real class are correctly identified.

F1 combines precision and recall into a single score.

Macro metrics weight each class equally.

Weighted metrics account for class support, so classes with more samples have greater influence.

## Output Files

The evaluation output directory contains:

```text
metrics.json
predictions.csv
per_class_metrics.csv
confusion_matrix.csv
confusion_matrix.png
misclassified.csv
confusion_pairs.csv
```

### `metrics.json`

Contains aggregate evaluation metrics together with checkpoint metadata such as:

- checkpoint epoch
- checkpoint path
- class mapping
- selected device

### `predictions.csv`

Contains one row for every test sample, including:

- relative image path
- true class
- predicted class
- confidence
- correctness

### `per_class_metrics.csv`

Contains precision, recall, F1, and support for each class.

### `confusion_matrix.csv`

Stores the raw-count confusion matrix using the checkpoint class order.

### `confusion_matrix.png`

Provides a visual representation of the confusion matrix.

### `misclassified.csv`

Contains incorrect predictions sorted by confidence, making high-confidence mistakes easier to inspect.

### `confusion_pairs.csv`

Summarizes the most frequent true-class to predicted-class confusion pairs.

The reports use relative image paths only and do not copy or redistribute raw images.

## Error Analysis

Start with `confusion_pairs.csv` to identify the most common class confusions.

Then inspect `misclassified.csv`, especially high-confidence errors, to understand whether mistakes may be related to:

- visually similar disease classes
- difficult or ambiguous samples
- image quality
- model limitations
- possible data issues

Error analysis is diagnostic. It should not be used to retroactively select a different checkpoint after the test set has been evaluated.

## Evaluation Results

The final local test evaluation produced:

| Metric | Result |
|---|---:|
| Accuracy | 92.07% |
| Macro F1 | 0.897 |
| Weighted F1 | 0.920 |

The evaluation produced 289 misclassified test samples.

The most frequent confusion was:

```text
Spider mites → Target Spot
```

with 69 errors.

These results provide the final held-out test estimate for the selected model and complement the validation metrics used during training and checkpoint selection.