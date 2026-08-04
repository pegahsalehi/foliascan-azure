# Azure ML Connection Check

Phase 2.6 adds a local, read-only connection check for the existing FoliaScan
Azure Machine Learning workspace. The command verifies that the Poetry
environment can authenticate and reach the workspace before any later Azure ML
training work is attempted.

Existing development resources:

- resource group: `rg-foliascan-dev-ne`
- workspace: `mlw-foliascan-dev-ne`
- compute targets: `cpu-fs-dev`, `gpu-fs-dev`

The subscription ID is intentionally not documented or hard-coded. Keep it in
your local environment only.

## Local Authentication

Authenticate locally with the Azure CLI:

```powershell
az login
```

If you have access to more than one subscription, select the correct one:

```powershell
az account set --subscription <your-subscription-id>
```

Then set the environment variables used by FoliaScan:

```powershell
$env:AZURE_SUBSCRIPTION_ID = "<your-subscription-id>"
$env:AZURE_RESOURCE_GROUP = "rg-foliascan-dev-ne"
$env:AZURE_ML_WORKSPACE = "mlw-foliascan-dev-ne"
```

## What The Command Uses

The verifier uses `azure.identity.DefaultAzureCredential`. Locally, that
credential can use your Azure CLI sign-in, along with other standard Azure SDK
credential sources.

It then creates an `azure.ai.ml.MLClient` for the configured subscription,
resource group, and workspace. Creating `MLClient` alone is not enough to prove
connectivity because construction is mostly local configuration. A real service
request is forced with:

```python
ml_client.workspaces.get(workspace_name)
```

After the workspace is reached, the command lists existing compute targets with:

```python
ml_client.compute.list()
```

Only read operations are used.

## Run The Check

```powershell
poetry run python -m foliascan.cloud.azure_connection
```

The command prints only:

- connection status
- workspace name
- workspace location
- compute name
- compute type
- provisioning state

It does not print subscription IDs, principal IDs, access tokens, or full
resource IDs.

## Safety

This command is read-only. It does not submit Azure ML jobs, start compute
nodes, upload data, or create, update, or delete Azure resources.

No Azure ML job or compute node is started by listing the workspace compute
targets.
