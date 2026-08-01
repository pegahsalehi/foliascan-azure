# Final Test Evaluation

Phase 1.6 adds final test-set evaluation and error-analysis tooling for the
local ResNet18 baseline. The training phase selected `best_model.pt` using only
training and validation records. The untouched test split remains reserved for
one final estimate of model quality.

Do not use test results repeatedly to tune transforms, hyperparameters, model
architecture, class handling, or checkpoint selection. Those choices belong to
training and validation. Once the test split influences model changes, it is no
longer an independent estimate.

## Command

Run evaluation manually after the leakage-safe manifest, exported images, and
selected checkpoint are available:

```powershell
poetry run python -m foliascan.evaluation.evaluate `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --checkpoint artifacts/training/resnet18_baseline_v1/best_model.pt `
  --output-dir artifacts/evaluation/resnet18_baseline_v1 `
  --device cuda
```

Use `--device cpu` for CPU evaluation, or `--device auto` to use CUDA only when
it is available. A non-empty output directory is rejected unless `--overwrite`
is passed.

## What Evaluation Does

Evaluation restores the checkpoint model architecture, weights, class mapping,
and image-size/batch-size settings needed for deterministic test inference. It
does not restore the optimizer, because evaluation does not update model
parameters.

Only manifest rows with `split=test` are loaded. The DataLoader uses the same
deterministic evaluation transform as validation: resize, center crop, tensor
conversion, and ImageNet normalization. The model runs with `model.eval()` and
`torch.inference_mode()`.

## Metrics

The evaluator reports:

- overall accuracy
- per-class precision, recall, F1, and support
- macro precision, recall, and F1
- weighted precision, recall, and F1
- a raw-count confusion matrix

Accuracy is the fraction of test samples predicted correctly. Precision answers
"when the model predicted this class, how often was it right?" Recall answers
"of the real samples in this class, how many did the model find?" F1 combines
precision and recall.

Macro metrics average classes equally. Weighted metrics average by class
support, so larger classes have more influence.

## Output Files

The output directory contains:

- `metrics.json`: aggregate metrics, checkpoint epoch, portable checkpoint path,
  class mapping, and selected device
- `predictions.csv`: every test prediction with path, true class, predicted
  class, confidence, and correctness
- `per_class_metrics.csv`: precision, recall, F1, and support per class
- `confusion_matrix.csv`: raw-count matrix in checkpoint class order
- `confusion_matrix.png`: readable matplotlib confusion-matrix image
- `misclassified.csv`: incorrect predictions sorted by confidence descending
- `confusion_pairs.csv`: most frequent true-class to predicted-class mistakes

The reports reference relative image paths only. They do not copy or
redistribute raw images.

## Error Analysis

Start with `confusion_pairs.csv` to identify the most common class confusions.
Then inspect `misclassified.csv`, especially high-confidence mistakes, to
separate likely model limitations from possible data issues.

Error analysis should be treated as diagnostic information for future phases,
not a reason to retroactively reselect the completed checkpoint.

## Later Phases

Explainability and Azure integration are later phases. This evaluator is a
local, manifest-driven final test pass only.
