"""Read-only Azure ML data asset verification for FoliaScan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
)

from foliascan.data.discovery import SUPPORTED_IMAGE_EXTENSIONS

REQUIRED_ENVIRONMENT_VARIABLES: Final[tuple[str, ...]] = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE",
)
EXPECTED_LOCAL_RECORD_COUNT: Final[int] = 18160
VERIFICATION_STATUS: Final[str] = "verified"


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


class AzureDataAssetVerificationError(ValueError):
    """Raised when Azure ML data asset verification cannot complete."""


class MissingAzureDataAssetConfigurationError(AzureDataAssetVerificationError):
    """Raised when required Azure environment variables are absent."""


class AzureDataAssetAuthenticationError(AzureDataAssetVerificationError):
    """Raised when local Azure authentication fails."""


class AzureDataAssetWorkspaceAccessError(AzureDataAssetVerificationError):
    """Raised when Azure ML workspace data assets cannot be accessed."""


class AzureDataAssetMissingError(AzureDataAssetVerificationError):
    """Raised when an expected Azure ML data asset is missing."""


class AzureDataAssetValidationError(AzureDataAssetVerificationError):
    """Raised when an Azure ML data asset does not match expectations."""


class LocalDataAssetError(AzureDataAssetVerificationError):
    """Raised when local data files cannot be verified."""


class LocalDataAssetMissingError(LocalDataAssetError):
    """Raised when an expected local data path is missing."""


class DataOperations(Protocol):
    """Azure ML data operations used by this read-only check."""

    def get(self, *, name: str, version: str) -> object:
        """Return one registered Azure ML data asset by name and version."""


class AzureMLClient(Protocol):
    """Minimal MLClient protocol for data asset verification."""

    data: DataOperations


@dataclass(frozen=True, slots=True)
class AzureDataAssetConfig:
    """Environment-derived Azure ML workspace configuration."""

    subscription_id: str
    resource_group: str
    workspace_name: str


@dataclass(frozen=True, slots=True)
class RequiredTag:
    """A required Azure ML data asset tag."""

    key: str
    value: str


@dataclass(frozen=True, slots=True)
class ExpectedAzureDataAsset:
    """Expected registered Azure ML data asset metadata."""

    name: str
    version: str
    asset_type: str
    required_tags: tuple[RequiredTag, ...]


@dataclass(frozen=True, slots=True)
class AzureDataAssetSummary:
    """Safe summary of one verified Azure ML data asset."""

    name: str
    version: str
    asset_type: str
    required_tags: tuple[RequiredTag, ...]
    cloud_path_present: bool


@dataclass(frozen=True, slots=True)
class ManifestFileSummary:
    """Local manifest row count and content hash."""

    role: str
    path: Path
    row_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AzureDataAssetVerificationSummary:
    """Result of read-only Azure ML data asset verification."""

    verification_status: str
    assets: tuple[AzureDataAssetSummary, ...]
    image_root: Path
    local_image_count: int
    dataset_manifest: ManifestFileSummary
    source_manifest: ManifestFileSummary
    expected_local_record_count: int


COMMON_REQUIRED_TAGS: Final[tuple[RequiredTag, ...]] = (
    RequiredTag("project", "foliascan"),
    RequiredTag("source", "plantvillage"),
)
EXPECTED_DATA_ASSETS: Final[tuple[ExpectedAzureDataAsset, ...]] = (
    ExpectedAzureDataAsset(
        name="foliascan-tomato-images",
        version="1",
        asset_type="uri_folder",
        required_tags=COMMON_REQUIRED_TAGS
        + (RequiredTag("content", "tomato-images"),),
    ),
    ExpectedAzureDataAsset(
        name="foliascan-dataset-manifest",
        version="1",
        asset_type="uri_file",
        required_tags=COMMON_REQUIRED_TAGS
        + (RequiredTag("content", "dataset-manifest"),),
    ),
    ExpectedAzureDataAsset(
        name="foliascan-source-manifest",
        version="1",
        asset_type="uri_file",
        required_tags=COMMON_REQUIRED_TAGS
        + (RequiredTag("content", "source-manifest"),),
    ),
)


def read_azure_data_asset_config(
    environ: Mapping[str, str] | None = None,
) -> AzureDataAssetConfig:
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
        raise MissingAzureDataAssetConfigurationError(msg)

    return AzureDataAssetConfig(
        subscription_id=values["AZURE_SUBSCRIPTION_ID"],
        resource_group=values["AZURE_RESOURCE_GROUP"],
        workspace_name=values["AZURE_ML_WORKSPACE"],
    )


def verify_azure_data_assets(
    *,
    image_root: Path,
    dataset_manifest: Path,
    source_manifest: Path,
    environ: Mapping[str, str] | None = None,
    expected_local_record_count: int = EXPECTED_LOCAL_RECORD_COUNT,
    expected_assets: Sequence[ExpectedAzureDataAsset] = EXPECTED_DATA_ASSETS,
) -> AzureDataAssetVerificationSummary:
    """Verify local data files and expected Azure ML data assets read-only."""

    config = read_azure_data_asset_config(environ)
    local_image_count = _count_local_images(image_root)
    _validate_local_count(
        role="local image root",
        path=image_root,
        actual_count=local_image_count,
        expected_count=expected_local_record_count,
    )
    dataset_manifest_summary = _manifest_file_summary(
        role="dataset_manifest",
        path=dataset_manifest,
        expected_row_count=expected_local_record_count,
    )
    source_manifest_summary = _manifest_file_summary(
        role="source_manifest",
        path=source_manifest,
        expected_row_count=expected_local_record_count,
    )

    try:
        credential = _default_azure_credential_factory()()
        ml_client = _create_ml_client(
            credential=credential,
            config=config,
        )
    except ClientAuthenticationError as exc:
        raise AzureDataAssetAuthenticationError(_authentication_message()) from exc
    except AzureError as exc:
        raise AzureDataAssetWorkspaceAccessError(_workspace_message()) from exc
    except ValueError as exc:
        raise AzureDataAssetWorkspaceAccessError(_workspace_message()) from exc

    asset_summaries = tuple(
        _fetch_and_validate_data_asset(ml_client, expected_asset)
        for expected_asset in expected_assets
    )
    return AzureDataAssetVerificationSummary(
        verification_status=VERIFICATION_STATUS,
        assets=asset_summaries,
        image_root=image_root,
        local_image_count=local_image_count,
        dataset_manifest=dataset_manifest_summary,
        source_manifest=source_manifest_summary,
        expected_local_record_count=expected_local_record_count,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Azure ML data asset verification CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        summary = verify_azure_data_assets(
            image_root=_namespace_path(args, "image_root"),
            dataset_manifest=_namespace_path(args, "dataset_manifest"),
            source_manifest=_namespace_path(args, "source_manifest"),
        )
    except AzureDataAssetVerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print_data_asset_summary(summary)
    return 0


def print_data_asset_summary(summary: AzureDataAssetVerificationSummary) -> None:
    """Print a concise, non-sensitive Azure ML data asset report."""

    print(f"verification_status: {summary.verification_status}")
    print(f"registered_assets: {len(summary.assets)}")
    for asset in summary.assets:
        cloud_path_status = "present" if asset.cloud_path_present else "missing"
        print(
            f"asset: {asset.name} version={asset.version} "
            f"type={asset.asset_type} cloud_path={cloud_path_status} tags=ok"
        )
    print(f"local_image_count: {summary.local_image_count}")
    print(f"dataset_manifest_rows: {summary.dataset_manifest.row_count}")
    print(f"dataset_manifest_sha256: {summary.dataset_manifest.sha256}")
    print(f"source_manifest_rows: {summary.source_manifest.row_count}")
    print(f"source_manifest_sha256: {summary.source_manifest.sha256}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m foliascan.cloud.azure_data_assets",
        description=(
            "Verify expected Azure ML data assets and matching local FoliaScan "
            "data files using read-only operations."
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Local directory containing the exported Tomato image class folders.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
        help="Local leakage-safe FoliaScan CSV manifest.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="Local PlantVillage source CSV manifest.",
    )
    return parser


def _default_azure_credential_factory() -> DefaultAzureCredentialFactory:
    if DefaultAzureCredential is not None:
        return DefaultAzureCredential
    try:
        from azure.identity import DefaultAzureCredential as credential_factory
    except ImportError as exc:
        raise AzureDataAssetVerificationError(_missing_sdk_message()) from exc
    return cast(DefaultAzureCredentialFactory, credential_factory)


def _ml_client_factory() -> MLClientFactory:
    if MLClient is not None:
        return MLClient
    try:
        from azure.ai.ml import MLClient as ml_client_factory
    except ImportError as exc:
        raise AzureDataAssetVerificationError(_missing_sdk_message()) from exc
    return cast(MLClientFactory, ml_client_factory)


def _create_ml_client(
    *,
    credential: object,
    config: AzureDataAssetConfig,
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


def _fetch_and_validate_data_asset(
    ml_client: AzureMLClient,
    expected: ExpectedAzureDataAsset,
) -> AzureDataAssetSummary:
    try:
        asset = ml_client.data.get(name=expected.name, version=expected.version)
    except ClientAuthenticationError as exc:
        raise AzureDataAssetAuthenticationError(_authentication_message()) from exc
    except ResourceNotFoundError as exc:
        msg = (
            "Missing Azure ML data asset: "
            f"{expected.name} version {expected.version}."
        )
        raise AzureDataAssetMissingError(msg) from exc
    except HttpResponseError as exc:
        msg = (
            "Azure ML data asset could not be accessed: "
            f"{expected.name} version {expected.version}. "
            "Confirm your account has read access to workspace data assets."
        )
        raise AzureDataAssetWorkspaceAccessError(msg) from exc
    except AzureError as exc:
        msg = (
            "Azure ML data asset could not be accessed: "
            f"{expected.name} version {expected.version}. "
            "Confirm your account has read access to workspace data assets."
        )
        raise AzureDataAssetWorkspaceAccessError(msg) from exc

    return _validate_data_asset(asset, expected)


def _validate_data_asset(
    asset: object,
    expected: ExpectedAzureDataAsset,
) -> AzureDataAssetSummary:
    actual_name = _object_text(asset, "name")
    if actual_name != expected.name:
        msg = (
            f"Azure ML data asset name mismatch for {expected.name}: "
            f"found {_display_text(actual_name)}."
        )
        raise AzureDataAssetValidationError(msg)

    actual_version = _object_text(asset, "version")
    if actual_version != expected.version:
        msg = (
            f"Azure ML data asset {expected.name} has wrong version: "
            f"expected {expected.version}, found {_display_text(actual_version)}."
        )
        raise AzureDataAssetValidationError(msg)

    actual_type = _object_text(asset, "type")
    if actual_type != expected.asset_type:
        msg = (
            f"Azure ML data asset {expected.name} has wrong type: "
            f"expected {expected.asset_type}, found {_display_text(actual_type)}."
        )
        raise AzureDataAssetValidationError(msg)

    tags = _asset_tags(asset)
    for required_tag in expected.required_tags:
        actual_value = _normalize_text(tags.get(required_tag.key))
        if actual_value != required_tag.value:
            msg = (
                f"Azure ML data asset {expected.name} does not match required "
                f"tag '{required_tag.key}'."
            )
            raise AzureDataAssetValidationError(msg)

    cloud_path = _object_text(asset, "path")
    if not cloud_path:
        msg = f"Azure ML data asset {expected.name} has no cloud path."
        raise AzureDataAssetValidationError(msg)

    return AzureDataAssetSummary(
        name=actual_name,
        version=actual_version,
        asset_type=actual_type,
        required_tags=expected.required_tags,
        cloud_path_present=True,
    )


def _count_local_images(image_root: Path) -> int:
    _ensure_local_directory(image_root, "image root")
    supported_extensions = frozenset(SUPPORTED_IMAGE_EXTENSIONS)
    image_count = 0
    try:
        for path in image_root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in supported_extensions
                and not _has_hidden_part(path.relative_to(image_root))
            ):
                image_count += 1
    except OSError as exc:
        msg = f"Could not read local image root: {_format_local_path(image_root)}"
        raise LocalDataAssetError(msg) from exc
    return image_count


def _manifest_file_summary(
    *,
    role: str,
    path: Path,
    expected_row_count: int,
) -> ManifestFileSummary:
    _ensure_local_file(path, role)
    row_count = _csv_row_count(path, role)
    _validate_local_count(
        role=role,
        path=path,
        actual_count=row_count,
        expected_count=expected_row_count,
    )
    return ManifestFileSummary(
        role=role,
        path=path,
        row_count=row_count,
        sha256=_sha256_file(path, role),
    )


def _csv_row_count(path: Path, role: str) -> int:
    try:
        with path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                msg = f"{role} is missing a CSV header: {_format_local_path(path)}"
                raise LocalDataAssetError(msg)
            return sum(1 for _ in reader)
    except OSError as exc:
        msg = f"Could not read {role}: {_format_local_path(path)}"
        raise LocalDataAssetError(msg) from exc


def _sha256_file(path: Path, role: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        msg = f"Could not hash {role}: {_format_local_path(path)}"
        raise LocalDataAssetError(msg) from exc
    return digest.hexdigest()


def _validate_local_count(
    *,
    role: str,
    path: Path,
    actual_count: int,
    expected_count: int,
) -> None:
    if actual_count != expected_count:
        msg = (
            f"Expected {expected_count} rows/items for {role}, "
            f"found {actual_count}: {_format_local_path(path)}"
        )
        raise LocalDataAssetError(msg)


def _ensure_local_directory(path: Path, role: str) -> None:
    try:
        exists = path.exists()
        is_dir = path.is_dir()
    except OSError as exc:
        msg = f"Could not inspect {role}: {_format_local_path(path)}"
        raise LocalDataAssetError(msg) from exc

    if not exists:
        msg = f"Missing {role} directory: {_format_local_path(path)}"
        raise LocalDataAssetMissingError(msg)
    if not is_dir:
        msg = f"{role} is not a directory: {_format_local_path(path)}"
        raise LocalDataAssetError(msg)


def _ensure_local_file(path: Path, role: str) -> None:
    try:
        exists = path.exists()
        is_file = path.is_file()
    except OSError as exc:
        msg = f"Could not inspect {role}: {_format_local_path(path)}"
        raise LocalDataAssetError(msg) from exc

    if not exists:
        msg = f"Missing {role} file: {_format_local_path(path)}"
        raise LocalDataAssetMissingError(msg)
    if not is_file:
        msg = f"{role} is not a file: {_format_local_path(path)}"
        raise LocalDataAssetError(msg)


def _asset_tags(asset: object) -> Mapping[str, object]:
    tags = getattr(asset, "tags", None)
    if isinstance(tags, Mapping):
        return cast(Mapping[str, object], tags)
    return {}


def _object_text(instance: object, attribute_name: str) -> str:
    return _normalize_text(getattr(instance, attribute_name, None))


def _normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value.strip()
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _display_text(value: str) -> str:
    return value or "missing"


def _has_hidden_part(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _format_local_path(path: Path) -> str:
    return path.as_posix()


def _namespace_path(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if isinstance(value, Path):
        return value
    msg = f"Expected path argument for {name}."
    raise TypeError(msg)


def _authentication_message() -> str:
    return (
        "Azure authentication failed. Run 'az login' and ensure the account has "
        "read access to the configured Azure ML workspace."
    )


def _workspace_message() -> str:
    return (
        "Azure ML workspace data assets could not be accessed. Check "
        "AZURE_RESOURCE_GROUP and AZURE_ML_WORKSPACE, and confirm your account "
        "has read access."
    )


def _missing_sdk_message() -> str:
    return (
        "Azure SDK dependencies are not installed in this Poetry environment. "
        "Run 'poetry install' before checking Azure ML data assets."
    )


if __name__ == "__main__":
    raise SystemExit(main())
