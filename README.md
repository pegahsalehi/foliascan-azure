# FoliaScan

FoliaScan is a learning and portfolio project for building a crop-leaf disease image-classification system. The project starts with a local Python implementation and is planned to grow into an Azure Machine Learning workflow for training, tracking, registering, and deploying a model.

This project was selected because plant health classification is a practical computer vision problem with a clear path from local experimentation to cloud-based machine learning operations. It offers room to practice dataset validation, baseline modeling, evaluation, explainability, deployment, and monitoring without adding unnecessary product complexity in the early phases.

FoliaScan is educational software. It is not a diagnostic tool. Any future predictions from this project must not replace advice from agricultural specialists, agronomists, plant pathologists, or other qualified professionals.

<p align="center">
  <img src="images/screenshot.png" alt="Project screenshot" width="600">
</p>

## Current Phase and Status

The project is in Phase 4.2: the existing training entry point is prepared for
Azure ML command-job mounted inputs and managed outputs. Phase 3, Azure ML data
asset registration and read-only verification, is complete. Phase 1B2,
official PlantVillage ingestion and leakage-safe split preparation, is
complete. Local baseline training and final local evaluation tooling are
available.

Current status:

- Phase 1A local project foundation is complete.
- Python package structure is initialized.
- Project-level path configuration is available in `src/foliascan/config.py`.
- Example training settings are documented in `configs/training.example.yaml`.
- Dataset preparation notes are documented in `docs/dataset.md`.
- Local training foundation notes are documented in `docs/training.md`.
- Final evaluation notes are documented in `docs/evaluation.md`.
- Azure ML connection-check notes are documented in `docs/azure-connection.md`.
- Azure ML data-asset verification notes are documented in
  `docs/azure-data-assets.md`.
- The planned official dataset source is `mohanty/PlantVillage` on Hugging Face,
  using the `color` configuration and the Tomato-only subset.
- Tests and quality-tool configuration are ready for local validation.
- No Azure training, deployment, web interface, or CI/CD has been implemented.

## Planned Workflow

The planned workflow is local first, then Azure:

1. Build and validate the project locally.
2. Prepare and validate the image dataset.
3. Train a baseline image classifier locally.
4. Evaluate model quality and add explainability.
5. Define Azure infrastructure only when the local workflow is understood.
6. Train with Azure Machine Learning and track experiments with MLflow.
7. Register the model and deploy it to a managed online endpoint.
8. Build a simple client application for inference.
9. Add GitHub Actions and monitoring.

## Planned Project Phases

1. Local project foundation
2. Dataset preparation and validation
3. Local baseline image classifier
4. Evaluation and explainability
5. Azure infrastructure
6. Azure Machine Learning training and MLflow tracking
7. Model registration and managed online endpoint
8. Simple client application
9. GitHub Actions and monitoring

## Installation

This project uses Python 3.11 and Poetry.

```powershell
poetry install
```

To run commands inside the Poetry environment:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

## Dataset Preparation

See `docs/dataset.md` for the official PlantVillage source, expected folder
structure, limitations, and local manifest workflow.

Export the official PlantVillage Tomato color subset:

```powershell
poetry run python -m foliascan.data.cli plantvillage-export `
  --output-dir data/raw/plantvillage_tomato_color `
  --source-manifest data/processed/plantvillage_source_manifest.csv
```

Create the leakage-safe FoliaScan manifest:

```powershell
poetry run python -m foliascan.data.cli plantvillage-split `
  --source-manifest data/processed/plantvillage_source_manifest.csv `
  --output data/processed/dataset_manifest.csv `
  --validation-ratio 0.15 `
  --random-seed 42
```

The generic split command below remains available for simple folder-based
datasets. It is not used for PlantVillage because PlantVillage requires
`leaf_id` group-aware splitting and preservation of the official test split.

Inspect a local directory-based dataset:

```powershell
poetry run python -m foliascan.data.cli inspect `
  --data-dir data/raw/plantvillage_tomato_color
```

Generate a stratified manifest from valid images:

```powershell
poetry run python -m foliascan.data.cli split `
  --data-dir data/raw/plantvillage_tomato_color `
  --output data/processed/dataset_manifest.csv
```

## Local Training Foundation

See `docs/training.md` for the manifest-driven PyTorch data pipeline, class
mapping, ResNet18 model factory, smoke-test workflow, and local training loop.

Run a forward-pass smoke test without training the model:

```powershell
poetry run python -m foliascan.training.smoke_test `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml `
  --split train
```

Train the local baseline without using the test split for model selection:

```powershell
poetry run python -m foliascan.training.train `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml
```

The same entry point accepts Azure ML mounted input paths, an Azure-managed
`--output-dir`, and optional `--max-train-batches` /
`--max-validation-batches` smoke-test limits.

## Final Test Evaluation

See `docs/evaluation.md` for the final test-set evaluation workflow, metrics,
confusion-matrix outputs, and error-analysis reports.

Run the selected checkpoint on the untouched test split:

```powershell
poetry run python -m foliascan.evaluation.evaluate `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --checkpoint artifacts/training/resnet18_baseline_v1/best_model.pt `
  --output-dir artifacts/evaluation/resnet18_baseline_v1 `
  --device cuda
```

Do not use final test results for repeated model tuning or checkpoint
selection.

## Azure ML Connection Check

See `docs/azure-connection.md` for the read-only local authentication and
workspace connectivity check.

Verify that the local Poetry environment can reach the existing Azure ML
workspace:

```powershell
poetry run python -m foliascan.cloud.azure_connection
```

The command reads `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, and
`AZURE_ML_WORKSPACE` from the local environment. It does not submit jobs, start
compute nodes, upload data, or create Azure resources.

## Azure ML Data Asset Check

See `docs/azure-data-assets.md` for the read-only registered data-asset and
local dataset verification workflow.

```powershell
poetry run python -m foliascan.cloud.azure_data_assets `
  --image-root data/raw/plantvillage_tomato_color `
  --dataset-manifest data/processed/dataset_manifest.csv `
  --source-manifest data/processed/plantvillage_source_manifest.csv
```

## Repository Structure

This repository is being prepared for the intended final project naming. The
GitHub repository and local folder renames remain manual steps.

```text
foliascan-azure/
|-- src/
|   `-- foliascan/
|       |-- __init__.py
|       |-- config.py
|       `-- data/
|-- tests/
|-- configs/
|-- docs/
|-- notebooks/
|-- data/
|-- sample_images/
|-- .env.example
|-- .gitignore
|-- AGENTS.md
|-- LICENSE
|-- pyproject.toml
`-- README.md
```

## Naming Conventions

- Display name: FoliaScan
- GitHub repository name: `foliascan-azure`
- Local folder name: `foliascan-azure`
- Poetry distribution name: `foliascan-azure`
- Python package name: `foliascan`

## Local Configuration

Use `.env.example` as a template for future local configuration. Do not commit real `.env` files, credentials, API keys, tokens, Azure subscription IDs, or other secrets.
