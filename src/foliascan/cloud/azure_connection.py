"""Read-only Azure ML workspace connection verification."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
)

REQUIRED_ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE",
)
CONNECTED_STATUS = "connected"


class DefaultAzureCredentialFactory(Protocol):
    """Factory protocol for Azure DefaultAzureCredential."""

    def __call__(self) -> object:
        """Create one Azure credential instance."""


class MLClientFactory(Protocol):
    """Factory protocol for Azure MLClient."""

    def __call__(
        self,
        *,
        credential: object,
        subscription_id: str,
        resource_group_name: str,
        workspace_name: str,
    ) -> object:
        """Create one Azure ML client instance."""


DefaultAzureCredential: DefaultAzureCredentialFactory | None = None
MLClient: MLClientFactory | None = None


class AzureConnectionError(ValueError):
    """Raised when Azure ML connection verification cannot complete."""


class MissingAzureConfigurationError(AzureConnectionError):
    """Raised when required Azure environment variables are absent."""


class AzureAuthenticationFailureError(AzureConnectionError):
    """Raised when local Azure authentication fails."""


class AzureWorkspaceAccessError(AzureConnectionError):
    """Raised when the configured Azure ML workspace cannot be accessed."""


class WorkspaceOperations(Protocol):
    """Azure ML workspace operations used by this read-only check."""

    def get(self, name: str) -> object:
        """Return one workspace by name."""


class ComputeOperations(Protocol):
    """Azure ML compute operations used by this read-only check."""

    def list(self) -> Iterable[object]:
        """Return existing compute targets."""


class AzureMLClient(Protocol):
    """Minimal MLClient protocol for connection verification."""

    workspaces: WorkspaceOperations
    compute: ComputeOperations


@dataclass(frozen=True, slots=True)
class AzureConnectionConfig:
    """Environment-derived Azure ML workspace configuration."""

    subscription_id: str
    resource_group: str
    workspace_name: str


@dataclass(frozen=True, slots=True)
class ComputeSummary:
    """Read-only Azure ML compute target summary."""

    name: str
    compute_type: str
    provisioning_state: str


@dataclass(frozen=True, slots=True)
class AzureConnectionSummary:
    """Result of Azure ML workspace connection verification."""

    connection_status: str
    workspace_name: str
    workspace_location: str
    compute_targets: tuple[ComputeSummary, ...]


def read_azure_connection_config(
    environ: Mapping[str, str] | None = None,
) -> AzureConnectionConfig:
    """Read required Azure ML workspace settings from environment variables."""

    environment = os.environ if environ is None else environ
    values: dict[str, str] = {}
    missing: list[str] = []
    for variable_name in REQUIRED_ENVIRONMENT_VARIABLES:
        value = environment.get(variable_name, "").strip()
        if not value:
            missing.append(variable_name)
        else:
            values[variable_name] = value

    if missing:
        msg = (
            "Missing required Azure configuration: "
            + ", ".join(sorted(missing))
        )
        raise MissingAzureConfigurationError(msg)

    return AzureConnectionConfig(
        subscription_id=values["AZURE_SUBSCRIPTION_ID"],
        resource_group=values["AZURE_RESOURCE_GROUP"],
        workspace_name=values["AZURE_ML_WORKSPACE"],
    )


def verify_azure_ml_connection(
    environ: Mapping[str, str] | None = None,
) -> AzureConnectionSummary:
    """Verify local authentication and read-only Azure ML workspace access."""

    config = read_azure_connection_config(environ)
    try:
        credential = _default_azure_credential_factory()()
        ml_client = _create_ml_client(
            credential=credential,
            config=config,
        )
    except ClientAuthenticationError as exc:
        raise AzureAuthenticationFailureError(_authentication_message()) from exc
    except AzureError as exc:
        raise AzureWorkspaceAccessError(_workspace_configuration_message()) from exc

    workspace = _get_workspace(ml_client, config.workspace_name)
    compute_targets = _list_compute_targets(ml_client)
    return AzureConnectionSummary(
        connection_status=CONNECTED_STATUS,
        workspace_name=_object_text(workspace, "name", fallback=config.workspace_name),
        workspace_location=_object_text(workspace, "location", fallback="unknown"),
        compute_targets=compute_targets,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Azure ML connection verification CLI."""

    parser = _build_parser()
    parser.parse_args(argv)

    try:
        summary = verify_azure_ml_connection()
    except AzureConnectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print_connection_summary(summary)
    return 0


def print_connection_summary(summary: AzureConnectionSummary) -> None:
    """Print the non-sensitive Azure ML connection summary."""

    print(f"connection_status: {summary.connection_status}")
    print(f"workspace_name: {summary.workspace_name}")
    print(f"workspace_location: {summary.workspace_location}")
    for compute_target in summary.compute_targets:
        print(f"compute_name: {compute_target.name}")
        print(f"compute_type: {compute_target.compute_type}")
        print(f"provisioning_state: {compute_target.provisioning_state}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m foliascan.cloud.azure_connection",
        description=(
            "Verify read-only authentication and connectivity to the configured "
            "Azure ML workspace."
        ),
    )
    return parser


def _default_azure_credential_factory() -> DefaultAzureCredentialFactory:
    if DefaultAzureCredential is not None:
        return DefaultAzureCredential
    try:
        from azure.identity import DefaultAzureCredential as credential_factory
    except ImportError as exc:
        raise AzureConnectionError(_missing_sdk_message()) from exc
    return cast(DefaultAzureCredentialFactory, credential_factory)


def _ml_client_factory() -> MLClientFactory:
    if MLClient is not None:
        return MLClient
    try:
        from azure.ai.ml import MLClient as ml_client_factory
    except ImportError as exc:
        raise AzureConnectionError(_missing_sdk_message()) from exc
    return cast(MLClientFactory, ml_client_factory)


def _create_ml_client(
    *,
    credential: object,
    config: AzureConnectionConfig,
) -> AzureMLClient:
    return cast(
        AzureMLClient,
        _ml_client_factory()(
            credential=credential,
            subscription_id=config.subscription_id,
            resource_group_name=config.resource_group,
            workspace_name=config.workspace_name,
        ),
    )


def _get_workspace(ml_client: AzureMLClient, workspace_name: str) -> object:
    try:
        return ml_client.workspaces.get(workspace_name)
    except ClientAuthenticationError as exc:
        raise AzureAuthenticationFailureError(_authentication_message()) from exc
    except ResourceNotFoundError as exc:
        raise AzureWorkspaceAccessError(_workspace_configuration_message()) from exc
    except HttpResponseError as exc:
        raise AzureWorkspaceAccessError(_workspace_configuration_message()) from exc


def _list_compute_targets(
    ml_client: AzureMLClient,
) -> tuple[ComputeSummary, ...]:
    try:
        return tuple(_compute_summary(compute) for compute in ml_client.compute.list())
    except ClientAuthenticationError as exc:
        raise AzureAuthenticationFailureError(_authentication_message()) from exc
    except HttpResponseError as exc:
        raise AzureWorkspaceAccessError(_compute_access_message()) from exc


def _compute_summary(compute: object) -> ComputeSummary:
    return ComputeSummary(
        name=_object_text(compute, "name", fallback="unknown"),
        compute_type=_object_text(
            compute,
            "type",
            fallback=_object_text(compute, "compute_type", fallback="unknown"),
        ),
        provisioning_state=_object_text(
            compute,
            "provisioning_state",
            fallback="unknown",
        ),
    )


def _object_text(instance: object, attribute_name: str, *, fallback: str) -> str:
    value = getattr(instance, attribute_name, None)
    if value is None:
        return fallback
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    return text or fallback


def _authentication_message() -> str:
    return (
        "Azure authentication failed. Run 'az login' and ensure the account has "
        "read access to the configured Azure ML workspace."
    )


def _workspace_configuration_message() -> str:
    return (
        "Azure ML workspace could not be accessed. Check AZURE_RESOURCE_GROUP "
        "and AZURE_ML_WORKSPACE, and confirm your account has read access."
    )


def _compute_access_message() -> str:
    return (
        "Azure ML workspace was reached, but compute targets could not be "
        "listed. Confirm your account has read access to workspace compute."
    )


def _missing_sdk_message() -> str:
    return (
        "Azure SDK dependencies are not installed in this Poetry environment. "
        "Run 'poetry install' before checking the Azure ML connection."
    )


if __name__ == "__main__":
    raise SystemExit(main())
