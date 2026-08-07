# Continuous Integration

FoliaScan uses GitHub Actions CI for repository-level validation. The workflow
is intentionally limited to deterministic local checks so pull requests can be
validated without Azure credentials, cloud compute, or deployment side effects.

The CI workflow has been successfully validated on a real pull request.

## Triggers

The workflow runs on:

- pull requests targeting `main`
- pushes to `main`

## Runner And Tooling

CI runs on a GitHub-hosted Ubuntu runner:

- runner: `ubuntu-latest`
- Python: `3.11`
- Poetry: `2.4.1`

Poetry is installed with `pipx`. Python is configured with
`actions/setup-python`, and Poetry dependency caching is enabled with
`poetry.lock` as the cache dependency path.

Dependencies are installed with:

```bash
poetry install --no-interaction --no-ansi
```

## Automated Checks

CI runs the same core checks used locally:

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy
poetry run python -c "import importlib; importlib.import_module('app.streamlit_app')"
```

The pytest step runs the full test suite. Ruff checks lint and import ordering.
mypy performs strict static type checking for the `foliascan` package. The
Streamlit import smoke test verifies that the app module can be imported without
making an Azure request.

## Permissions

The workflow uses read-only repository permissions:

```yaml
permissions:
  contents: read
```

## Azure Scope

CI intentionally excludes Azure operations. It does not:

- authenticate to Azure
- require Azure secrets
- create Azure resources
- submit Azure ML jobs
- run Azure training
- deploy managed online endpoints
- call a real Azure ML endpoint

Keeping CI local-only prevents accidental Azure compute cost and makes automated
validation predictable for pull requests and pushes to `main`.
