# Azure ML Connection Check

FoliaScan includes a small read-only utility for verifying local access to the Azure Machine Learning workspace.

The check confirms that the local Poetry environment can authenticate to Azure, reach the configured workspace, and list the available compute targets before cloud jobs are submitted.

Development resources:

- Resource group: `rg-foliascan-dev-ne`
- Workspace: `mlw-foliascan-dev-ne`
- Compute targets: `cpu-fs-dev`, `gpu-fs-dev`

The Azure subscription ID is not hard-coded or stored in the repository.

## Authentication

Authenticate with the Azure CLI:

```powershell
az login
```

If more than one subscription is available, select the required one:

```powershell
az account set --subscription <your-subscription-id>
```

Set the environment variables used by FoliaScan:

```powershell
$env:AZURE_SUBSCRIPTION_ID = "<your-subscription-id>"
$env:AZURE_RESOURCE_GROUP = "rg-foliascan-dev-ne"
$env:AZURE_ML_WORKSPACE = "mlw-foliascan-dev-ne"
```

## How It Works

The connection check uses:

```text
azure.identity.DefaultAzureCredential
```

Locally, this can use the active Azure CLI sign-in.

An Azure ML client is then created with:

```text
azure.ai.ml.MLClient
```

Creating the client alone does not confirm connectivity, so the utility performs a real read request:

```python
ml_client.workspaces.get(workspace_name)
```

It then lists the compute targets with:

```python
ml_client.compute.list()
```

Only read operations are performed.

## Run the Check

```powershell
poetry run python -m foliascan.cloud.azure_connection
```

The output contains only operational information such as:

- connection status
- workspace name
- workspace location
- compute name
- compute type
- provisioning state

Sensitive values such as subscription IDs, access tokens, principal IDs, and full Azure resource IDs are not printed.

## Safety

The connection check is read-only.

It does not:

- submit Azure ML jobs
- start compute nodes
- upload data
- create resources
- modify resources
- delete resources

Listing workspace compute targets does not start the compute clusters.