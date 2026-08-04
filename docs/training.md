# Local Training Foundation

Phase 2A added the local data and model foundation for a simple image
classification baseline. Phase 2B adds local training and validation for that
baseline. Phase 4.2 keeps the same entry point usable when Azure ML command
jobs provide mounted input and output paths. It does not submit Azure jobs,
calculate final test accuracy, register models, deploy inference endpoints, or
add MLflow tracking.

## Baseline Model

FoliaScan starts with ResNet18 because it is a well-known, compact convolutional
baseline with predictable torchvision support. It is strong enough to validate
the data path and model shape without adding architecture complexity too early.

Transfer learning is supported through the `pretrained` setting. When
`freeze_backbone` is enabled, FoliaScan freezes the ResNet feature extractor and
keeps the final classification layer trainable for the project-specific tomato
classes.

## Manifest-Driven Loading

Training data is read from the leakage-safe FoliaScan manifest:

```text
relative_path,class_name,split,leaf_id,source_split
```

The loader preserves `relative_path` as a `Path`, rejects absolute or duplicate
paths, validates split names, and loads images lazily with Pillow. Images are
converted to RGB before tensor conversion.

Class indexes are derived from the manifest by sorting class names
deterministically. The same mapping is used for train, validation, and test, and
numeric PlantVillage label IDs are not hard-coded.

## Transforms

Training transforms resize images, apply a conservative random crop, horizontal
flip, and small rotation, then use ImageNet normalization. The augmentation is
intentionally restrained because leaf orientation and disease texture can carry
important signal.

Validation and test transforms are deterministic: resize, center crop, tensor
conversion, and the same ImageNet normalization.

## Split Integrity

The leakage-safe splits created in Phase 1B2 must remain unchanged. The training
pipeline filters records by the manifest `split` column and does not resplit
images. The original `source_split` and `leaf_id` fields remain available for
auditability.

## Smoke Test

Run one forward-pass smoke test after the local PlantVillage export and
leakage-safe manifest exist:

```powershell
poetry run python -m foliascan.training.smoke_test `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml `
  --split train
```

The smoke test loads one batch, builds ResNet18, runs a single forward pass, and
checks that the output shape is `batch_size x number_of_classes`. It uses CPU by
default unless an available device is explicitly requested. It does not train
the model.

## Local Training

Run local baseline training with:

```powershell
poetry run python -m foliascan.training.train `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml
```

For a short integration check, override the epoch count:

```powershell
poetry run python -m foliascan.training.train `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml `
  --epochs 1 `
  --output-dir artifacts/training/one_epoch_check
```

For a tiny smoke-test training pass, limit the number of batches processed per
epoch:

```powershell
poetry run python -m foliascan.training.train `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml `
  --epochs 1 `
  --max-train-batches 2 `
  --max-validation-batches 1 `
  --output-dir artifacts/training/batch_limit_check
```

The batch limits must be positive integers when supplied. Omitting them
processes every batch. The limits cap only processed batches; they do not alter
the dataset manifest or change train, validation, or test split assignments.

The training loop uses `torch.nn.CrossEntropyLoss` for multi-class
classification and AdamW for optimization. AdamW starts as the only supported
optimizer because it is a practical, stable default for transfer-learning
baselines and keeps the first local training phase simple.

Training records update model weights. Validation records guide model selection,
checkpointing, and early stopping. Test records are not loaded by the training
loop and must not be used for model selection; final test evaluation is a later
phase.

## Azure ML Command-Job Inputs

Azure ML command jobs can mount registered data assets into job-specific paths.
For FoliaScan, the expected registered assets are:

- `foliascan-tomato-images:1` as a `uri_folder`
- `foliascan-dataset-manifest:1` as a `uri_file`
- `foliascan-source-manifest:1` as a `uri_file`

Phase 4.2 reuses `python -m foliascan.training.train` instead of adding a
second training program. Reusing the existing entry point keeps the model,
dataset, DataLoader, engine, checkpoint, and orchestration logic in one place
for local and future Azure runs.

The training command accepts arbitrary mounted paths for:

- `--data-dir`: mounted image directory
- `--manifest`: mounted FoliaScan dataset manifest file
- `--config`: local YAML file included with the submitted code
- `--output-dir`: Azure-managed output directory

The path arguments are used as supplied. They are not resolved relative to
`PROJECT_ROOT`, so Linux mount paths such as `/mnt/azureml/...` work when the
same command runs inside an Azure ML job container.

An Azure-style command would have this shape inside the job:

```bash
python -m foliascan.training.train \
  --manifest /mnt/azureml/inputs/manifest/dataset_manifest.csv \
  --data-dir /mnt/azureml/inputs/images \
  --config configs/training.example.yaml \
  --output-dir /mnt/azureml/outputs/model
```

This repository phase prepares the entry point only. It does not submit the
Azure ML job, create compute, upload data, start compute, register a model, or
add MLflow. MLflow integration is a later subphase.

## Artifacts

Training writes the configured output directory only when training starts. An
existing empty output directory is accepted because Azure ML may provide an
empty mounted output path. A non-empty output directory is rejected unless
`--overwrite` is passed.

Each run writes:

- `history.csv`
- `history.json`
- `best_model.pt`
- `last_model.pt`

`best_model.pt` is selected by lowest validation loss. `last_model.pt` is
updated after every completed epoch. Checkpoints include model and optimizer
state, class mapping, the resolved training configuration, train and validation
metrics, best validation loss, and the random seed.

## Early Stopping

`early_stopping_patience` counts completed epochs without validation-loss
improvement. A value of `0` disables early stopping. Best and last checkpoints
are still written when early stopping occurs.

## Reproducibility

FoliaScan seeds Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA when CUDA
is available. CUDA requests are explicit: `device: auto` uses CUDA when
available and CPU otherwise; `device: cuda` fails clearly when CUDA is not
available. Deterministic cuDNN settings are enabled when CUDA is present, but
full bit-for-bit reproducibility can still depend on hardware, drivers, and
library versions.

MLflow tracking and Azure training are future phases.
