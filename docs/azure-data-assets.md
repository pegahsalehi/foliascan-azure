# Azure ML Data Asset Verification

FoliaScan includes a read-only utility for verifying the Azure Machine Learning data assets used by the project.

The check confirms that the expected registered assets exist in the configured workspace and that their metadata remains consistent with the local dataset files.

The Azure subscription ID is not hard-coded or stored in the repository.

## Registered Assets

The verifier retrieves each asset with:

```python
MLClient.data.get()
```

and validates its version, type, tags, and cloud-path availability.

| Name | Version | Type | Required tags |
| --- | --- | --- | --- |
| `foliascan-tomato-images` | `1` | `uri_folder` | `project=foliascan`, `source=plantvillage`, `content=tomato-images` |
| `foliascan-dataset-manifest` | `1` | `uri_file` | `project=foliascan`, `source=plantvillage`, `content=dataset-manifest` |
| `foliascan-source-manifest` | `1` | `uri_file` | `project=foliascan`, `source=plantvillage`, `content=source-manifest` |

The verifier confirms that a cloud path exists for each asset without printing the full storage location.

## Local Integrity Checks

Before connecting to Azure ML, the verifier checks the corresponding local files.

Expected values:

```text
Tomato images:          18,160
Dataset manifest rows:  18,160
Source manifest rows:   18,160
```

SHA-256 hashes are also calculated for both manifest files so their local contents can be compared and traced reliably.

## Authentication

Authenticate with Azure CLI when needed:

```powershell
az login
```

Then configure the workspace:

```powershell
$env:AZURE_SUBSCRIPTION_ID = "<your-subscription-id>"
$env:AZURE_RESOURCE_GROUP = "rg-foliascan-dev-ne"
$env:AZURE_ML_WORKSPACE = "mlw-foliascan-dev-ne"
```

## Run the Verification

```powershell
poetry run python -m foliascan.cloud.azure_data_assets `
  --image-root data/raw/plantvillage_tomato_color `
  --dataset-manifest data/processed/dataset_manifest.csv `
  --source-manifest data/processed/plantvillage_source_manifest.csv
```

The report includes:

- verification status
- asset name
- asset version
- asset type
- cloud-path availability
- local image and manifest counts
- manifest hashes

It does not print subscription IDs, access tokens, full Azure resource IDs, or full storage paths.

## Safety

The verification command is read-only.

It does not:

- upload or register data
- modify data assets
- archive or delete assets
- submit Azure ML jobs
- start compute
- create Azure resources

Its purpose is only to confirm that the registered data assets and local dataset metadata are still consistent.