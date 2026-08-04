import csv
import hashlib
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError

from foliascan.cloud import azure_data_assets
from foliascan.cloud.azure_data_assets import (
    AzureDataAssetAuthenticationError,
    AzureDataAssetMissingError,
    AzureDataAssetSummary,
    AzureDataAssetValidationError,
    AzureDataAssetVerificationSummary,
    LocalDataAssetError,
    LocalDataAssetMissingError,
    ManifestFileSummary,
    MissingAzureDataAssetConfigurationError,
    RequiredTag,
    read_azure_data_asset_config,
    verify_azure_data_assets,
)

SENSITIVE_SUBSCRIPTION_ID = "11111111-2222-3333-4444-555555555555"
SENSITIVE_RESOURCE_ID = (
    "/subscriptions/11111111-2222-3333-4444-555555555555/"
    "resourceGroups/rg-foliascan-dev-ne/providers/Microsoft.MachineLearningServices/"
    "workspaces/mlw-foliascan-dev-ne/data/foliascan-tomato-images/versions/1"
)
SENSITIVE_STORAGE_PATH = (
    "azureml://subscriptions/11111111-2222-3333-4444-555555555555/"
    "resourcegroups/rg-foliascan-dev-ne/workspaces/mlw-foliascan-dev-ne/"
    "datastores/workspaceblobstore/paths/foliascan/private/path"
)


class FakeCredential:
    def __init__(self) -> None:
        self.created = True


class FakeDataAsset:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        asset_type: str,
        tags: Mapping[str, str] | None = None,
        path: str | None = SENSITIVE_STORAGE_PATH,
    ) -> None:
        self.name = name
        self.version = version
        self.type = asset_type
        self.tags = dict(tags or _required_tags_for(name))
        self.path = path
        self.id = SENSITIVE_RESOURCE_ID


class FakeDataOperations:
    def __init__(
        self,
        calls: list[str],
        assets: Mapping[tuple[str, str], FakeDataAsset],
        failure: BaseException | None,
    ) -> None:
        self._calls = calls
        self._assets = assets
        self._failure = failure

    def get(self, *, name: str, version: str) -> FakeDataAsset:
        self._calls.append(f"data.get:{name}:{version}")
        if self._failure is not None:
            raise self._failure
        try:
            return self._assets[(name, version)]
        except KeyError as exc:
            raise ResourceNotFoundError(
                message=f"missing {SENSITIVE_RESOURCE_ID}",
            ) from exc

    def create_or_update(self, data: object) -> object:
        self._calls.append("data.create_or_update")
        raise AssertionError("data mutation must not be called")

    def archive(self, name: str, version: str) -> object:
        self._calls.append("data.archive")
        raise AssertionError("data archive must not be called")

    def delete(self, name: str, version: str) -> object:
        self._calls.append("data.delete")
        raise AssertionError("data deletion must not be called")


class FakeComputeOperations:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def list(self) -> object:
        self._calls.append("compute.list")
        raise AssertionError("compute must not be listed")


class FakeJobOperations:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def create_or_update(self, job: object) -> object:
        self._calls.append("jobs.create_or_update")
        raise AssertionError("jobs must not be submitted")


class FakeMLClient:
    instances: list["FakeMLClient"] = []
    assets: Mapping[tuple[str, str], FakeDataAsset] = {}
    failure: BaseException | None = None

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
        self.data = FakeDataOperations(self.calls, self.assets, self.failure)
        self.compute = FakeComputeOperations(self.calls)
        self.jobs = FakeJobOperations(self.calls)
        FakeMLClient.instances.append(self)


def test_missing_environment_variables_are_reported_without_values() -> None:
    with pytest.raises(MissingAzureDataAssetConfigurationError) as exc_info:
        read_azure_data_asset_config({})

    message = str(exc_info.value)
    assert "AZURE_SUBSCRIPTION_ID" in message
    assert "AZURE_RESOURCE_GROUP" in message
    assert "AZURE_ML_WORKSPACE" in message
    assert SENSITIVE_SUBSCRIPTION_ID not in message


def test_successful_data_asset_verification_is_read_only_and_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_root, dataset_manifest, source_manifest = _make_local_inputs(tmp_path, 3)
    _install_successful_azure_mocks(monkeypatch)

    summary = verify_azure_data_assets(
        image_root=image_root,
        dataset_manifest=dataset_manifest,
        source_manifest=source_manifest,
        environ=_environment(),
        expected_local_record_count=3,
    )
    azure_data_assets.print_data_asset_summary(summary)

    assert summary == AzureDataAssetVerificationSummary(
        verification_status="verified",
        assets=(
            AzureDataAssetSummary(
                name="foliascan-tomato-images",
                version="1",
                asset_type="uri_folder",
                required_tags=(
                    RequiredTag("project", "foliascan"),
                    RequiredTag("source", "plantvillage"),
                    RequiredTag("content", "tomato-images"),
                ),
                cloud_path_present=True,
            ),
            AzureDataAssetSummary(
                name="foliascan-dataset-manifest",
                version="1",
                asset_type="uri_file",
                required_tags=(
                    RequiredTag("project", "foliascan"),
                    RequiredTag("source", "plantvillage"),
                    RequiredTag("content", "dataset-manifest"),
                ),
                cloud_path_present=True,
            ),
            AzureDataAssetSummary(
                name="foliascan-source-manifest",
                version="1",
                asset_type="uri_file",
                required_tags=(
                    RequiredTag("project", "foliascan"),
                    RequiredTag("source", "plantvillage"),
                    RequiredTag("content", "source-manifest"),
                ),
                cloud_path_present=True,
            ),
        ),
        image_root=image_root,
        local_image_count=3,
        dataset_manifest=ManifestFileSummary(
            role="dataset_manifest",
            path=dataset_manifest,
            row_count=3,
            sha256=_sha256(dataset_manifest),
        ),
        source_manifest=ManifestFileSummary(
            role="source_manifest",
            path=source_manifest,
            row_count=3,
            sha256=_sha256(source_manifest),
        ),
        expected_local_record_count=3,
    )
    client = FakeMLClient.instances[-1]
    assert isinstance(client.credential, FakeCredential)
    assert client.subscription_id == SENSITIVE_SUBSCRIPTION_ID
    assert client.resource_group_name == "rg-foliascan-dev-ne"
    assert client.workspace_name == "mlw-foliascan-dev-ne"
    assert client.calls == [
        "data.get:foliascan-tomato-images:1",
        "data.get:foliascan-dataset-manifest:1",
        "data.get:foliascan-source-manifest:1",
    ]

    with pytest.raises(FrozenInstanceError):
        summary.local_image_count = 0  # type: ignore[misc]

    printed = capsys.readouterr().out
    assert "verification_status: verified" in printed
    assert "asset: foliascan-tomato-images version=1 type=uri_folder" in printed
    assert "dataset_manifest_sha256:" in printed
    assert "source_manifest_sha256:" in printed
    assert SENSITIVE_SUBSCRIPTION_ID not in printed
    assert SENSITIVE_RESOURCE_ID not in printed
    assert SENSITIVE_STORAGE_PATH not in printed
    assert "token" not in printed.lower()


def test_missing_azure_asset_is_reported_without_sensitive_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_root, dataset_manifest, source_manifest = _make_local_inputs(tmp_path, 1)
    _install_successful_azure_mocks(monkeypatch, assets={})

    with pytest.raises(AzureDataAssetMissingError) as exc_info:
        verify_azure_data_assets(
            image_root=image_root,
            dataset_manifest=dataset_manifest,
            source_manifest=source_manifest,
            environ=_environment(),
            expected_local_record_count=1,
        )

    message = str(exc_info.value)
    assert "foliascan-tomato-images version 1" in message
    assert SENSITIVE_SUBSCRIPTION_ID not in message
    assert SENSITIVE_RESOURCE_ID not in message
    assert FakeMLClient.instances[-1].calls == [
        "data.get:foliascan-tomato-images:1",
    ]


def test_authentication_failure_is_concise_and_non_sensitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_root, dataset_manifest, source_manifest = _make_local_inputs(tmp_path, 1)
    failure = ClientAuthenticationError(
        message=f"token failed for {SENSITIVE_SUBSCRIPTION_ID}",
    )
    _install_successful_azure_mocks(monkeypatch, failure=failure)

    with pytest.raises(AzureDataAssetAuthenticationError) as exc_info:
        verify_azure_data_assets(
            image_root=image_root,
            dataset_manifest=dataset_manifest,
            source_manifest=source_manifest,
            environ=_environment(),
            expected_local_record_count=1,
        )

    message = str(exc_info.value)
    assert "Azure authentication failed" in message
    assert SENSITIVE_SUBSCRIPTION_ID not in message


def test_wrong_asset_version_type_tags_and_missing_path_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_root, dataset_manifest, source_manifest = _make_local_inputs(tmp_path, 1)

    invalid_assets = (
        (
            FakeDataAsset(
                name="foliascan-tomato-images",
                version="2",
                asset_type="uri_folder",
            ),
            "wrong version",
        ),
        (
            FakeDataAsset(
                name="foliascan-tomato-images",
                version="1",
                asset_type="uri_file",
            ),
            "wrong type",
        ),
        (
            FakeDataAsset(
                name="foliascan-tomato-images",
                version="1",
                asset_type="uri_folder",
                tags={"project": "foliascan", "source": "plantvillage"},
            ),
            "required tag 'content'",
        ),
        (
            FakeDataAsset(
                name="foliascan-tomato-images",
                version="1",
                asset_type="uri_folder",
                path=None,
            ),
            "no cloud path",
        ),
    )

    for invalid_asset, expected_message in invalid_assets:
        _install_successful_azure_mocks(
            monkeypatch,
            assets={("foliascan-tomato-images", "1"): invalid_asset},
        )

        with pytest.raises(AzureDataAssetValidationError, match=expected_message):
            verify_azure_data_assets(
                image_root=image_root,
                dataset_manifest=dataset_manifest,
                source_manifest=source_manifest,
                environ=_environment(),
                expected_local_record_count=1,
            )


def test_missing_local_files_are_reported_before_azure_client_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_azure_mocks(monkeypatch)

    with pytest.raises(LocalDataAssetMissingError, match="Missing image root"):
        verify_azure_data_assets(
            image_root=tmp_path / "missing-images",
            dataset_manifest=tmp_path / "missing-dataset.csv",
            source_manifest=tmp_path / "missing-source.csv",
            environ=_environment(),
            expected_local_record_count=1,
        )

    assert FakeMLClient.instances == []


def test_local_manifest_row_counts_must_match_image_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_root, dataset_manifest, source_manifest = _make_local_inputs(tmp_path, 2)
    _write_dataset_manifest(dataset_manifest, 1)
    _install_successful_azure_mocks(monkeypatch)

    with pytest.raises(LocalDataAssetError) as exc_info:
        verify_azure_data_assets(
            image_root=image_root,
            dataset_manifest=dataset_manifest,
            source_manifest=source_manifest,
            environ=_environment(),
            expected_local_record_count=2,
        )

    assert "dataset_manifest" in str(exc_info.value)
    assert "Expected 2 rows/items" in str(exc_info.value)
    assert FakeMLClient.instances == []


def test_cli_returns_non_zero_for_expected_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_verify_azure_data_assets(**kwargs: object) -> object:
        raise LocalDataAssetMissingError("missing local file")

    monkeypatch.setattr(
        azure_data_assets,
        "verify_azure_data_assets",
        fake_verify_azure_data_assets,
    )

    exit_status = azure_data_assets.main(
        [
            "--image-root",
            "data/raw/plantvillage_tomato_color",
            "--dataset-manifest",
            "data/processed/dataset_manifest.csv",
            "--source-manifest",
            "data/processed/plantvillage_source_manifest.csv",
        ]
    )

    captured = capsys.readouterr()
    assert exit_status == 2
    assert "error: missing local file" in captured.err
    assert "Traceback" not in captured.err


def _install_successful_azure_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    assets: Mapping[tuple[str, str], FakeDataAsset] | None = None,
    failure: BaseException | None = None,
) -> None:
    FakeMLClient.instances.clear()
    FakeMLClient.assets = dict(_successful_assets() if assets is None else assets)
    FakeMLClient.failure = failure
    monkeypatch.setattr(azure_data_assets, "DefaultAzureCredential", FakeCredential)
    monkeypatch.setattr(azure_data_assets, "MLClient", FakeMLClient)


def _successful_assets() -> dict[tuple[str, str], FakeDataAsset]:
    return {
        ("foliascan-tomato-images", "1"): FakeDataAsset(
            name="foliascan-tomato-images",
            version="1",
            asset_type="uri_folder",
        ),
        ("foliascan-dataset-manifest", "1"): FakeDataAsset(
            name="foliascan-dataset-manifest",
            version="1",
            asset_type="uri_file",
        ),
        ("foliascan-source-manifest", "1"): FakeDataAsset(
            name="foliascan-source-manifest",
            version="1",
            asset_type="uri_file",
        ),
    }


def _required_tags_for(asset_name: str) -> dict[str, str]:
    content_by_name = {
        "foliascan-tomato-images": "tomato-images",
        "foliascan-dataset-manifest": "dataset-manifest",
        "foliascan-source-manifest": "source-manifest",
    }
    return {
        "project": "foliascan",
        "source": "plantvillage",
        "content": content_by_name[asset_name],
    }


def _make_local_inputs(root: Path, row_count: int) -> tuple[Path, Path, Path]:
    image_root = root / "raw" / "plantvillage_tomato_color"
    class_dir = image_root / "Tomato___healthy"
    class_dir.mkdir(parents=True)
    for index in range(row_count):
        (class_dir / f"image_{index:03}.jpg").touch()

    dataset_manifest = root / "processed" / "dataset_manifest.csv"
    source_manifest = root / "processed" / "plantvillage_source_manifest.csv"
    _write_dataset_manifest(dataset_manifest, row_count)
    _write_source_manifest(source_manifest, row_count)
    return image_root, dataset_manifest, source_manifest


def _write_dataset_manifest(path: Path, row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=(
                "relative_path",
                "class_name",
                "split",
                "leaf_id",
                "source_split",
            ),
        )
        writer.writeheader()
        for index in range(row_count):
            writer.writerow(
                {
                    "relative_path": f"Tomato___healthy/image_{index:03}.jpg",
                    "class_name": "Tomato___healthy",
                    "split": "train",
                    "leaf_id": f"leaf-{index:03}",
                    "source_split": "train",
                }
            )


def _write_source_manifest(path: Path, row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("relative_path", "class_name", "source_split", "leaf_id"),
        )
        writer.writeheader()
        for index in range(row_count):
            writer.writerow(
                {
                    "relative_path": f"Tomato___healthy/image_{index:03}.jpg",
                    "class_name": "Tomato___healthy",
                    "source_split": "train",
                    "leaf_id": f"leaf-{index:03}",
                }
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, str]:
    return {
        "AZURE_SUBSCRIPTION_ID": SENSITIVE_SUBSCRIPTION_ID,
        "AZURE_RESOURCE_GROUP": "rg-foliascan-dev-ne",
        "AZURE_ML_WORKSPACE": "mlw-foliascan-dev-ne",
    }
