from collections.abc import Mapping

import pytest
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from foliascan.cloud import azure_connection
from foliascan.cloud.azure_connection import (
    AzureAuthenticationFailureError,
    AzureConnectionSummary,
    AzureWorkspaceAccessError,
    MissingAzureConfigurationError,
    read_azure_connection_config,
    verify_azure_ml_connection,
)

SENSITIVE_SUBSCRIPTION_ID = "11111111-2222-3333-4444-555555555555"
SENSITIVE_RESOURCE_ID = (
    "/subscriptions/11111111-2222-3333-4444-555555555555/"
    "resourceGroups/rg-foliascan-dev-ne/providers/Microsoft.MachineLearningServices/"
    "workspaces/mlw-foliascan-dev-ne"
)


class FakeCredential:
    def __init__(self) -> None:
        self.created = True


class FakeWorkspace:
    name = "mlw-foliascan-dev-ne"
    location = "norwayeast"
    id = SENSITIVE_RESOURCE_ID


class FakeCompute:
    def __init__(
        self,
        *,
        name: str,
        compute_type: str,
        provisioning_state: str,
    ) -> None:
        self.name = name
        self.type = compute_type
        self.provisioning_state = provisioning_state
        self.id = SENSITIVE_RESOURCE_ID + f"/computes/{name}"


class FakeWorkspaceOperations:
    def __init__(self, calls: list[str], failure: BaseException | None = None) -> None:
        self._calls = calls
        self._failure = failure

    def get(self, name: str) -> FakeWorkspace:
        self._calls.append(f"workspaces.get:{name}")
        if self._failure is not None:
            raise self._failure
        return FakeWorkspace()

    def create_or_update(self, workspace: object) -> object:
        self._calls.append("workspaces.create_or_update")
        raise AssertionError("workspace mutation must not be called")

    def delete(self, name: str) -> object:
        self._calls.append("workspaces.delete")
        raise AssertionError("workspace deletion must not be called")


class FakeComputeOperations:
    def __init__(self, calls: list[str], failure: BaseException | None = None) -> None:
        self._calls = calls
        self._failure = failure

    def list(self) -> tuple[FakeCompute, ...]:
        self._calls.append("compute.list")
        if self._failure is not None:
            raise self._failure
        return (
            FakeCompute(
                name="cpu-fs-dev",
                compute_type="amlcompute",
                provisioning_state="Succeeded",
            ),
            FakeCompute(
                name="gpu-fs-dev",
                compute_type="amlcompute",
                provisioning_state="Stopped",
            ),
        )

    def begin_create_or_update(self, compute: object) -> object:
        self._calls.append("compute.begin_create_or_update")
        raise AssertionError("compute mutation must not be called")

    def delete(self, name: str) -> object:
        self._calls.append("compute.delete")
        raise AssertionError("compute deletion must not be called")


class FakeJobOperations:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def create_or_update(self, job: object) -> object:
        self._calls.append("jobs.create_or_update")
        raise AssertionError("job submission must not be called")


class FakeMLClient:
    instances: list["FakeMLClient"] = []

    def __init__(
        self,
        *,
        credential: object,
        subscription_id: str,
        resource_group_name: str,
        workspace_name: str,
    ) -> None:
        self.credential = credential
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        self.workspace_name = workspace_name
        self.calls: list[str] = []
        self.workspaces = FakeWorkspaceOperations(self.calls)
        self.compute = FakeComputeOperations(self.calls)
        self.jobs = FakeJobOperations(self.calls)
        FakeMLClient.instances.append(self)


def test_missing_environment_variables_are_reported_without_values() -> None:
    with pytest.raises(MissingAzureConfigurationError) as exc_info:
        read_azure_connection_config({})

    message = str(exc_info.value)
    assert "AZURE_SUBSCRIPTION_ID" in message
    assert "AZURE_RESOURCE_GROUP" in message
    assert "AZURE_ML_WORKSPACE" in message
    assert SENSITIVE_SUBSCRIPTION_ID not in message


def test_successful_workspace_access_and_compute_discovery(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_successful_azure_mocks(monkeypatch)

    summary = verify_azure_ml_connection(_environment())
    azure_connection.print_connection_summary(summary)

    assert summary == AzureConnectionSummary(
        connection_status="connected",
        workspace_name="mlw-foliascan-dev-ne",
        workspace_location="norwayeast",
        compute_targets=(
            azure_connection.ComputeSummary(
                name="cpu-fs-dev",
                compute_type="amlcompute",
                provisioning_state="Succeeded",
            ),
            azure_connection.ComputeSummary(
                name="gpu-fs-dev",
                compute_type="amlcompute",
                provisioning_state="Stopped",
            ),
        ),
    )
    client = FakeMLClient.instances[-1]
    assert isinstance(client.credential, FakeCredential)
    assert client.subscription_id == SENSITIVE_SUBSCRIPTION_ID
    assert client.resource_group_name == "rg-foliascan-dev-ne"
    assert client.workspace_name == "mlw-foliascan-dev-ne"
    assert client.calls == [
        "workspaces.get:mlw-foliascan-dev-ne",
        "compute.list",
    ]

    printed = capsys.readouterr().out
    assert "connection_status: connected" in printed
    assert "workspace_name: mlw-foliascan-dev-ne" in printed
    assert "workspace_location: norwayeast" in printed
    assert "compute_name: cpu-fs-dev" in printed
    assert "compute_name: gpu-fs-dev" in printed
    assert SENSITIVE_SUBSCRIPTION_ID not in printed
    assert SENSITIVE_RESOURCE_ID not in printed
    assert "rg-foliascan-dev-ne" not in printed
    assert "principal" not in printed.lower()
    assert "token" not in printed.lower()


def test_authentication_failure_is_concise_and_non_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    failure = ClientAuthenticationError(
        message=f"auth failed for {SENSITIVE_SUBSCRIPTION_ID}",
    )
    _install_successful_azure_mocks(monkeypatch)

    class AuthenticationFailingMLClient(FakeMLClient):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.calls = calls
            self.workspaces = FakeWorkspaceOperations(self.calls, failure=failure)
            self.compute = FakeComputeOperations(self.calls)

    monkeypatch.setattr(azure_connection, "MLClient", AuthenticationFailingMLClient)

    with pytest.raises(AzureAuthenticationFailureError) as exc_info:
        verify_azure_ml_connection(_environment())

    message = str(exc_info.value)
    assert "Azure authentication failed" in message
    assert SENSITIVE_SUBSCRIPTION_ID not in message
    assert calls == ["workspaces.get:mlw-foliascan-dev-ne"]


def test_inaccessible_workspace_is_reported_without_resource_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    failure = HttpResponseError(message=f"missing {SENSITIVE_RESOURCE_ID}")
    _install_successful_azure_mocks(monkeypatch)

    class WorkspaceFailingMLClient(FakeMLClient):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.calls = calls
            self.workspaces = FakeWorkspaceOperations(self.calls, failure=failure)
            self.compute = FakeComputeOperations(self.calls)

    monkeypatch.setattr(azure_connection, "MLClient", WorkspaceFailingMLClient)

    with pytest.raises(AzureWorkspaceAccessError) as exc_info:
        verify_azure_ml_connection(_environment())

    message = str(exc_info.value)
    assert "Azure ML workspace could not be accessed" in message
    assert SENSITIVE_SUBSCRIPTION_ID not in message
    assert SENSITIVE_RESOURCE_ID not in message
    assert calls == ["workspaces.get:mlw-foliascan-dev-ne"]


def test_cli_returns_non_zero_for_expected_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_verify_azure_ml_connection(
        environ: Mapping[str, str] | None = None,
    ) -> AzureConnectionSummary:
        raise MissingAzureConfigurationError("missing config")

    monkeypatch.setattr(
        azure_connection,
        "verify_azure_ml_connection",
        fake_verify_azure_ml_connection,
    )

    exit_status = azure_connection.main([])

    captured = capsys.readouterr()
    assert exit_status == 2
    assert "error: missing config" in captured.err
    assert "Traceback" not in captured.err


def test_no_create_update_delete_or_job_operation_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_azure_mocks(monkeypatch)

    verify_azure_ml_connection(_environment())

    client = FakeMLClient.instances[-1]
    assert client.calls == [
        "workspaces.get:mlw-foliascan-dev-ne",
        "compute.list",
    ]
    forbidden_operations = (
        "create",
        "update",
        "delete",
        "begin_create_or_update",
        "jobs.create_or_update",
    )
    assert not any(
        forbidden in call
        for call in client.calls
        for forbidden in forbidden_operations
    )


def _install_successful_azure_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeMLClient.instances.clear()
    monkeypatch.setattr(azure_connection, "DefaultAzureCredential", FakeCredential)
    monkeypatch.setattr(azure_connection, "MLClient", FakeMLClient)


def _environment() -> dict[str, str]:
    return {
        "AZURE_SUBSCRIPTION_ID": SENSITIVE_SUBSCRIPTION_ID,
        "AZURE_RESOURCE_GROUP": "rg-foliascan-dev-ne",
        "AZURE_ML_WORKSPACE": "mlw-foliascan-dev-ne",
    }
