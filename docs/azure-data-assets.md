# Azure ML Data Asset Verification

Phase 3.4 adds a local, read-only verification command for the FoliaScan Azure
Machine Learning data assets. It checks that the expected registered assets are
available in the configured workspace and still match the local dataset files.

The subscription ID is intentionally not documented or hard-coded. Keep it in
your local environment only.

## Expected Assets

The verifier fetches each asset with `MLClient.data.get()` and validates the
registered metadata:

| Name | Version | Type | Required tags |
| --- | --- | --- | --- |
| `foliascan-tomato-images` | `1` | `uri_folder` | `project=foliascan`, `source=plantvillage`, `content=tomato-images` |
| `foliascan-dataset-manifest` | `1` | `uri_file` | `project=foliascan`, `source=plantvillage`, `content=dataset-manifest` |
| `foliascan-source-manifest` | `1` | `uri_file` | `project=foliascan`, `source=plantvillage`, `content=source-manifest` |

Each asset must also expose a cloud path. The command only reports that a cloud
path is present; it does not print the full storage path.

## Local Checks

The verifier checks the local files before creating an Azure ML client:

- `data/raw/plantvillage_tomato_color` contains exactly `18160` supported image
  files.
- `data/processed/dataset_manifest.csv` contains exactly `18160` data rows.
- `data/processed/plantvillage_source_manifest.csv` contains exactly `18160`
  data rows.
- SHA-256 hashes are calculated for both local manifest files.

## Environment

Authenticate locally with the Azure CLI if needed:

```powershell
az login
```

Then set the environment variables used by the verifier:

```powershell
$env:AZURE_SUBSCRIPTION_ID = "<your-subscription-id>"
$env:AZURE_RESOURCE_GROUP = "rg-foliascan-dev-ne"
$env:AZURE_ML_WORKSPACE = "mlw-foliascan-dev-ne"
```

## Run The Check

```powershell
poetry run python -m foliascan.cloud.azure_data_assets `
  --image-root data/raw/plantvillage_tomato_color `
  --dataset-manifest data/processed/dataset_manifest.csv `
  --source-manifest data/processed/plantvillage_source_manifest.csv
```

The report includes only the verification status, asset names, versions, types,
whether each cloud path is present, local counts, and manifest hashes. It does
not print subscription IDs, full Azure resource IDs, access tokens, or full
storage paths.

## Safety

This command is read-only. It does not upload, create, update, archive, or
delete data assets. It does not submit Azure ML jobs, start compute nodes, or
create Azure resources.
