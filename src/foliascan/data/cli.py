"""Command-line interface for local dataset preparation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from foliascan.data.discovery import (
    DatasetDiscoveryError,
    discover_class_names,
    discover_image_records,
)
from foliascan.data.plantvillage import (
    PlantVillageError,
    create_and_write_leaf_group_manifest,
    export_tomato_subset,
    manifest_counts_by_split,
)
from foliascan.data.splitting import (
    create_split_assignments,
    split_counts,
    write_manifest,
)
from foliascan.data.validation import (
    DatasetSummary,
    summarize_dataset,
    validate_images,
    write_summary_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset CLI and return a process exit status."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "split":
            return _run_split(args)
        if args.command == "plantvillage-export":
            return _run_plantvillage_export(args)
        if args.command == "plantvillage-split":
            return _run_plantvillage_split(args)
    except (
        DatasetDiscoveryError,
        FileExistsError,
        OSError,
        PlantVillageError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m foliascan.data.cli",
        description="Inspect and split a directory-based image dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Discover and validate images under a dataset root.",
    )
    inspect_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per class.",
    )
    inspect_parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional path for a JSON inspection report.",
    )

    split_parser = subparsers.add_parser(
        "split",
        help="Create a stratified train/validation/test CSV manifest.",
    )
    split_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per class.",
    )
    split_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV manifest output path.",
    )
    split_parser.add_argument("--train-ratio", type=float, default=0.7)
    split_parser.add_argument("--validation-ratio", type=float, default=0.15)
    split_parser.add_argument("--test-ratio", type=float, default=0.15)
    split_parser.add_argument("--random-seed", type=int, default=42)
    split_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing manifest file.",
    )

    export_parser = subparsers.add_parser(
        "plantvillage-export",
        help="Download and export the official PlantVillage Tomato color subset.",
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for exported Tomato class folders.",
    )
    export_parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="CSV source manifest output path.",
    )
    export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing exported images and source manifest.",
    )

    plantvillage_split_parser = subparsers.add_parser(
        "plantvillage-split",
        help="Create a leakage-safe FoliaScan manifest from a source manifest.",
    )
    plantvillage_split_parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="CSV manifest created by plantvillage-export.",
    )
    plantvillage_split_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV FoliaScan dataset manifest output path.",
    )
    plantvillage_split_parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Validation ratio applied only to official training leaf groups.",
    )
    plantvillage_split_parser.add_argument("--random-seed", type=int, default=42)
    plantvillage_split_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing FoliaScan manifest file.",
    )

    return parser


def _run_inspect(args: argparse.Namespace) -> int:
    data_dir = _namespace_path(args, "data_dir")
    class_names = discover_class_names(data_dir)
    records = discover_image_records(data_dir)
    validation_results = validate_images(records)
    summary = summarize_dataset(records, validation_results, class_names)

    _print_summary(summary)

    json_report = _namespace_optional_path(args, "json_report")
    if json_report is not None:
        write_summary_report(summary, json_report, data_dir)
        print(f"JSON report: {json_report}")

    return 0


def _run_split(args: argparse.Namespace) -> int:
    data_dir = _namespace_path(args, "data_dir")
    records = discover_image_records(data_dir)
    validation_results = validate_images(records)
    valid_records = tuple(
        result.record for result in validation_results if result.is_valid
    )
    split_records = create_split_assignments(
        records=valid_records,
        dataset_root=data_dir,
        train_ratio=_namespace_float(args, "train_ratio"),
        validation_ratio=_namespace_float(args, "validation_ratio"),
        test_ratio=_namespace_float(args, "test_ratio"),
        random_seed=_namespace_int(args, "random_seed"),
    )
    output_path = _namespace_path(args, "output")
    write_manifest(
        split_records=split_records,
        manifest_path=output_path,
        overwrite=_namespace_bool(args, "overwrite"),
    )

    print(f"Manifest written: {output_path}")
    for count_record in split_counts(split_records):
        print(f"{count_record.key}: {count_record.count}")

    return 0


def _run_plantvillage_export(args: argparse.Namespace) -> int:
    output_dir = _namespace_path(args, "output_dir")
    source_manifest = _namespace_path(args, "source_manifest")

    print(
        "Warning: this command downloads the official PlantVillage color dataset "
        "from Hugging Face and may download multiple gigabytes."
    )
    summary = export_tomato_subset(
        output_dir=output_dir,
        source_manifest_path=source_manifest,
        overwrite=_namespace_bool(args, "overwrite"),
    )

    print("PlantVillage export complete")
    print(f"Dataset: {summary.dataset_id} ({summary.dataset_config})")
    print(f"Output directory: {summary.output_dir}")
    print(f"Source manifest: {summary.source_manifest_path}")
    print(f"Image format: {summary.image_format}")
    print(f"Tomato records exported: {summary.tomato_records_exported}")
    for source_split, count in summary.source_split_counts:
        print(f"official {source_split}: {count}")

    return 0


def _run_plantvillage_split(args: argparse.Namespace) -> int:
    source_manifest = _namespace_path(args, "source_manifest")
    output_path = _namespace_path(args, "output")
    records = create_and_write_leaf_group_manifest(
        source_manifest_path=source_manifest,
        output_path=output_path,
        validation_ratio=_namespace_float(args, "validation_ratio"),
        random_seed=_namespace_int(args, "random_seed"),
        overwrite=_namespace_bool(args, "overwrite"),
    )

    print("PlantVillage leaf-group split complete")
    print(f"Source manifest: {source_manifest}")
    print(f"FoliaScan manifest: {output_path}")
    for split_name, count in manifest_counts_by_split(records):
        print(f"{split_name}: {count}")

    return 0


def _print_summary(summary: DatasetSummary) -> None:
    imbalance = (
        f"{summary.class_imbalance_ratio:.3f}"
        if summary.class_imbalance_ratio is not None
        else "undefined"
    )
    print("Dataset inspection complete")
    print(f"Classes: {summary.total_class_count}")
    print(f"Discovered images: {summary.total_discovered_images}")
    print(f"Valid images: {summary.total_valid_images}")
    print(f"Corrupted images: {summary.total_corrupted_images}")
    print(f"Smallest class size: {summary.smallest_class_size}")
    print(f"Largest class size: {summary.largest_class_size}")
    print(f"Class imbalance ratio: {imbalance}")


def _namespace_path(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if isinstance(value, Path):
        return value
    msg = f"Expected path argument for {name}."
    raise TypeError(msg)


def _namespace_optional_path(args: argparse.Namespace, name: str) -> Path | None:
    value = getattr(args, name)
    if value is None or isinstance(value, Path):
        return value
    msg = f"Expected optional path argument for {name}."
    raise TypeError(msg)


def _namespace_float(args: argparse.Namespace, name: str) -> float:
    value = getattr(args, name)
    if isinstance(value, float):
        return value
    msg = f"Expected float argument for {name}."
    raise TypeError(msg)


def _namespace_int(args: argparse.Namespace, name: str) -> int:
    value = getattr(args, name)
    if isinstance(value, int):
        return value
    msg = f"Expected integer argument for {name}."
    raise TypeError(msg)


def _namespace_bool(args: argparse.Namespace, name: str) -> bool:
    value = getattr(args, name)
    if isinstance(value, bool):
        return value
    msg = f"Expected boolean argument for {name}."
    raise TypeError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
