# Continuous Integration

FoliaScan uses GitHub Actions to automatically validate code quality on pull requests and after changes are merged into `main`.

The workflow runs entirely on GitHub-hosted infrastructure and does not require Azure credentials or cloud resources.

## Triggers

CI runs on:

- pull requests targeting `main`
- pushes to `main`

This validates changes before merge and verifies the final state of the main branch afterward.

## Environment

The workflow uses:

```text
Runner:  ubuntu-latest
Python:  3.11
Poetry:  2.4.1
```

Dependencies are installed with:

```bash
poetry install --no-interaction --no-ansi
```

Poetry dependency caching is enabled using `poetry.lock` as the cache dependency path.

## Automated Checks

The workflow runs:

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy
poetry run python -c "import importlib; importlib.import_module('app.streamlit_app')"
```

These checks cover:

- the full automated test suite
- Ruff linting
- mypy static type checking
- a Streamlit import smoke test

The Streamlit smoke test also confirms that importing the application does not trigger a real Azure request.

## Permissions

The workflow uses read-only repository access:

```yaml
permissions:
  contents: read
```

No write permission is required for CI.

## Azure Boundary

Azure operations are intentionally kept outside the CI workflow.

CI does not:

- authenticate to Azure
- use Azure secrets
- create Azure resources
- submit training jobs
- start cloud compute
- deploy managed online endpoints
- call a real Azure ML endpoint

This keeps routine validation predictable and avoids accidental cloud usage or cost.

## Validation

The workflow was tested through real pull requests and successfully ran both before merge and after changes reached `main`.