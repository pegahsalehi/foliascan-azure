# Training

FoliaScan uses a single PyTorch training pipeline for both local development and Azure Machine Learning jobs. The same entry point handles dataset loading, augmentation, training, validation, checkpointing, artifact generation, and optional MLflow tracking.

The model is a ResNet18 classifier trained on the leakage-safe PlantVillage Tomato dataset manifest.

## Model

FoliaScan uses ResNet18 as a compact transfer-learning baseline.

When pretrained weights are enabled, the model starts from ImageNet features and replaces the original classification layer with a ten-class output layer for the tomato-condition classes.

The training configuration also supports freezing the backbone while keeping the final classification layer trainable.

## Dataset Loading

Training data is read from the FoliaScan manifest:

```text
relative_path,class_name,split,leaf_id,source_split
```

The loader:

- validates manifest paths and split names
- rejects absolute and duplicate paths
- loads images lazily with Pillow
- converts images to RGB
- derives class indexes deterministically from sorted class names

The same class mapping is used for training, validation, testing, and inference.

The existing manifest split assignments are preserved. The training pipeline does not resplit the data, which keeps the `leaf_id`-based leakage protection intact.

## Image Transforms

Training images use light augmentation:

- resize
- conservative random crop
- horizontal flip
- small rotation
- ImageNet normalization

Validation and test transforms are deterministic:

- resize
- center crop
- tensor conversion
- ImageNet normalization

Random augmentation is used only during training.

## Smoke Test

Before running a full training job, the pipeline can be checked with a single forward pass:

```powershell
poetry run python -m foliascan.training.smoke_test `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml `
  --split train
```

The smoke test loads one batch, builds the model, runs inference, and verifies that the output shape matches:

```text
batch_size × number_of_classes
```

It does not update model parameters.

## Local Training

Run the full local training pipeline with:

```powershell
poetry run python -m foliascan.training.train `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml
```

For a short integration check:

```powershell
poetry run python -m foliascan.training.train `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml `
  --epochs 1 `
  --output-dir artifacts/training/one_epoch_check
```

Training and validation batches can also be limited for lightweight pipeline checks:

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

The batch limits affect only how many batches are processed. They do not change the dataset or its split assignments.

## Training Behaviour

The training loop uses:

- `torch.nn.CrossEntropyLoss`
- AdamW optimization
- configurable learning rate and weight decay
- training and validation loops
- checkpoint selection by validation loss
- optional early stopping

Training records update model weights.

Validation records are used for model selection, checkpointing, and early stopping.

The test split is not used during training or checkpoint selection.

## Checkpoints and Artifacts

Each completed training run writes:

```text
history.csv
history.json
best_model.pt
last_model.pt
```

`best_model.pt` stores the checkpoint with the lowest validation loss.

`last_model.pt` stores the most recently completed epoch.

The checkpoints contain the information needed to reconstruct and trace the training run, including:

- model state
- optimizer state
- class mapping
- resolved training configuration
- train and validation metrics
- best validation loss
- random seed

A non-empty output directory is rejected unless `--overwrite` is explicitly supplied.

## Early Stopping

`early_stopping_patience` controls how many completed epochs without validation-loss improvement are allowed before training stops.

A value of `0` disables early stopping.

The best and last checkpoints are still written when early stopping occurs.

## Azure Machine Learning

The same training entry point is used for Azure ML command jobs:

```text
python -m foliascan.training.train
```

Azure ML supplies mounted paths for registered data assets and a managed output directory. The training code accepts those paths directly instead of assuming that data exists relative to the local repository.

The registered training inputs are:

- `foliascan-tomato-images:1` — `uri_folder`
- `foliascan-dataset-manifest:1` — `uri_file`
- `foliascan-source-manifest:1` — `uri_file`

The main path arguments are:

- `--data-dir` — image directory
- `--manifest` — FoliaScan dataset manifest
- `--config` — training configuration
- `--output-dir` — training artifact directory

An Azure-style command has this form:

```bash
python -m foliascan.training.train \
  --manifest /mnt/azureml/inputs/manifest/dataset_manifest.csv \
  --data-dir /mnt/azureml/inputs/images \
  --config configs/training.example.yaml \
  --output-dir /mnt/azureml/outputs/model \
  --enable-mlflow
```

Using one training entry point keeps local and cloud execution aligned and avoids maintaining separate implementations.

## MLflow Tracking

MLflow tracking is optional and enabled with:

```text
--enable-mlflow
```

Inside an Azure ML command job, the training code uses the run context provided by Azure ML. It does not hard-code a tracking URI or create a separate MLflow experiment.

Tracked parameters include:

- model name
- epochs
- batch size
- learning rate
- optimizer
- weight decay
- image size
- pretrained and backbone-freezing settings
- random seed
- requested device
- optional batch limits

Metrics are logged after each epoch:

- training loss
- training accuracy
- validation loss
- validation accuracy
- learning rate
- elapsed time

The epoch number is used as the MLflow step so metric histories can be compared across runs.

Final summary metrics include:

- completed epochs
- best epoch
- best validation loss
- best validation accuracy
- early-stopping status

MLflow stores lightweight artifacts:

- training configuration
- `history.csv`
- `history.json`

The large checkpoint files remain in the Azure ML named training output rather than being duplicated in MLflow.

Experiment tracking and model registration are kept separate: MLflow records how a run behaved, while the selected checkpoint is registered independently as a versioned Azure ML model asset.

## Reproducibility

FoliaScan seeds:

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA when available

CUDA behaviour is explicit:

```text
device: auto  → use CUDA when available, otherwise CPU
device: cuda  → require CUDA and fail if it is unavailable
```

Deterministic cuDNN settings are enabled when CUDA is available.

Exact bit-for-bit reproducibility can still depend on hardware, drivers, and library versions.