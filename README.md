<p align="center">
  <img src="assets/branding/foliascan-logo.png" alt="FoliaScan logo" width="300">
</p>

# FoliaScan

FoliaScan is an educational and portfolio-focused end-to-end Azure Machine
Learning computer vision project for tomato-leaf image classification. It shows
the path from dataset preparation and local ResNet18 training through Azure ML
training, model registration, managed online endpoint validation, and a small
Streamlit inference client.

FoliaScan is not a professional agricultural diagnosis tool. Its predictions
are educational model outputs and should not replace advice from agronomists,
plant pathologists, extension services, or other qualified specialists.

<p align="center">
  <img src="assets/screenshots/streamlit-cloud-inference.png" alt="FoliaScan Streamlit cloud inference screenshot" width="400">
</p>

## Overview

The project uses the Tomato subset of PlantVillage to practice a realistic
machine learning workflow without turning the repository into a production
product. It emphasizes leakage-safe data splitting, reproducible training,
honest final test evaluation, Azure ML orchestration, secure endpoint
configuration, and a focused user-facing client.

The Phase 8 cloud endpoint was validated with real Streamlit-to-Azure inference
and then deleted to control Azure cost. The repository keeps the code,
configuration, tests, and documentation needed to recreate a compatible
endpoint deliberately.

## Key Features

- Leakage-safe PlantVillage Tomato dataset preparation.
- Manifest-driven PyTorch data loading and ResNet18 baseline training.
- Final local test evaluation with metrics, confusion matrix, and error
  analysis outputs.
- Azure ML infrastructure definitions for workspace, compute, environments,
  jobs, data assets, model registration, and managed online endpoint deployment.
- Azure GPU training with MLflow experiment tracking.
- Registered model asset and tested Azure ML Managed Online Endpoint contract.
- Local Docker deployment validation for the scoring script.
- Streamlit client application for one-image cloud inference.
- Mocked tests for endpoint invocation and presentation helpers.

## End-To-End Architecture

```text
PlantVillage Tomato images
        |
        v
Leakage-safe FoliaScan manifest
        |
        v
Local / Azure ML ResNet18 training
        |
        v
Final evaluation and model registration
        |
        v
Azure ML Managed Online Endpoint
        |
        v
Streamlit upload -> HTTPS request -> prediction display
```

The Streamlit app sends a raw Base64 JPEG or PNG payload to the managed online
endpoint and displays the returned class probabilities. It does not use Azure
SDK packages for inference; endpoint invocation is ordinary HTTPS.

## Model / Evaluation Results

Azure GPU training selected the best checkpoint at epoch 10:

- validation loss: `0.212930`
- validation accuracy: `0.928964`
- final training accuracy: `0.941544`

The separate local final test evaluation reported:

- accuracy: `92.07%`
- macro F1: `0.89665`
- weighted F1: `0.92042`

Validation metrics and final test metrics come from different stages. The
validation split guided checkpoint selection during training; the final test
split was reserved for the completed model evaluation.

See [docs/evaluation.md](docs/evaluation.md) for the final evaluation workflow
and output files.

## Azure ML Workflow

FoliaScan includes Azure ML configuration for:

- read-only workspace connectivity checks
- versioned data asset verification
- CPU smoke jobs
- GPU training jobs
- MLflow run tracking
- inference environment definition
- registered model asset metadata
- managed online endpoint and deployment YAML
- local endpoint deployment validation

Azure resource creation and job submission are deliberate manual actions. The
repository does not create cloud resources during tests or app import.

Relevant docs:

- [Azure connection check](docs/azure-connection.md)
- [Azure data assets](docs/azure-data-assets.md)
- [Training workflow](docs/training.md)

## Streamlit Application

The Streamlit app lives in [app/streamlit_app.py](app/streamlit_app.py). It
supports one simple workflow:

```text
upload image -> preview -> analyze -> prediction
```

It displays the predicted tomato-leaf class, confidence, top three class
probabilities, and an expandable table of all class probabilities. Backend class
names such as `Tomato___Late_blight` are formatted as readable labels such as
`Late Blight` in the UI only; the endpoint contract is unchanged.

Client details are documented in
[docs/client-application.md](docs/client-application.md).

## Learning Roadmap

<p align="center">
  <img src="assets/roadmap/foliascan-learning-roadmap.png" alt="FoliaScan learning roadmap" width="760">
</p>

The roadmap documents the phase-by-phase Azure ML learning journey behind this
project, from local foundations through cloud inference.

- [Download the PDF learning notes](assets/roadmap/foliascan-learning-roadmap.pdf)
- [Download the editable XMind roadmap](assets/roadmap/foliascan-learning-roadmap.xmind)

## Learning Materials

- [Dataset preparation](docs/dataset.md)
- [Training workflow](docs/training.md)
- [Evaluation workflow](docs/evaluation.md)
- [Azure connection check](docs/azure-connection.md)
- [Azure data asset verification](docs/azure-data-assets.md)
- [Client application](docs/client-application.md)

## Project Phases

Completed work includes:

1. Local project foundation.
2. Leakage-safe PlantVillage Tomato dataset preparation.
3. Local ResNet18 baseline training.
4. Final evaluation and error analysis.
5. Azure ML infrastructure configuration.
6. Versioned Azure data assets.
7. CPU smoke job and GPU model training in Azure.
8. MLflow experiment tracking.
9. Registered model asset.
10. Azure ML Managed Online Endpoint contract and deployment configuration.
11. Local Docker deployment validation.
12. Successful real cloud inference.
13. Streamlit client application.
14. Successful end-to-end Streamlit to Azure endpoint inference.

Possible later work includes CI/CD, monitoring, richer explainability, and
field-image robustness studies.

## Repository Structure

```text
foliascan-azure/
|-- app/
|   `-- streamlit_app.py
|-- assets/
|   |-- branding/
|   |-- roadmap/
|   `-- screenshots/
|-- configs/
|-- docs/
|-- infra/
|   `-- azure/
|-- sample_images/
|-- src/
|   `-- foliascan/
|       |-- client/
|       |-- cloud/
|       |-- data/
|       |-- evaluation/
|       |-- inference/
|       `-- training/
|-- tests/
|-- AGENTS.md
|-- LICENSE
|-- poetry.lock
|-- pyproject.toml
`-- README.md
```

## Installation / Local Setup

This project uses Python 3.11 and Poetry.

```powershell
poetry install
```

Common local checks:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy
```

Dataset export requires the Hugging Face PlantVillage source and can be large.
See [docs/dataset.md](docs/dataset.md) before downloading or preparing data.

## Running The Streamlit App

From the repository root:

```powershell
poetry run streamlit run app/streamlit_app.py
```

The app can import and render without endpoint configuration. A prediction
requires a compatible deployed Azure ML Managed Online Endpoint and the
environment variables below.

## Azure Configuration

The Streamlit client reads:

```powershell
$env:FOLIASCAN_ENDPOINT_URL = "<managed-online-endpoint-scoring-url>"
$env:FOLIASCAN_ENDPOINT_KEY = "<managed-online-endpoint-key>"
```

Do not commit real endpoint URLs, keys, subscription identifiers, resource
identifiers, tokens, or `.env` files. The app does not display these values.

Azure ML training and data-asset commands use their own documented environment
variables. Keep those values local.

## Security And Cost Notes

- Tests mock HTTP calls and do not call Azure.
- App import does not call Azure.
- The endpoint key is read from the environment and never hard-coded.
- The Streamlit app does not store uploaded images, predictions, or history.
- Azure resources should be created, deployed, and deleted deliberately.
- The Phase 8 cloud endpoint was deleted after validation to avoid ongoing
  managed endpoint cost.

## Disclaimer

FoliaScan is educational software built to practice computer vision and Azure ML
workflows. Plant disease recognition can be affected by image quality, cultivar,
lighting, disease stage, mixed symptoms, field conditions, and dataset bias. Do
not use FoliaScan as the sole basis for agricultural treatment decisions.
