import csv
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from foliascan.data import cli
from foliascan.data.plantvillage import (
    DATASET_CONFIG,
    DATASET_ID,
    HUGGINGFACE_SCRIPT_BUILDER_CONFIG,
    HUGGINGFACE_SCRIPT_FILENAME,
    FoliaScanManifestRecord,
    PlantVillageError,
    PlantVillageExportSummary,
    SourceManifestRecord,
    create_leaf_group_manifest,
    export_tomato_subset,
    load_official_plantvillage_dataset,
    normalize_plantvillage_records,
    read_source_manifest,
    validate_leaf_group_manifest,
    write_foliascan_manifest,
)

EXPECTED_CLASSES = ("Tomato___Bacterial_spot", "Tomato___healthy")


def test_load_official_plantvillage_dataset_downloads_official_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_dataset = object()
    downloaded_script_path = str(tmp_path / HUGGINGFACE_SCRIPT_FILENAME)
    hub_calls: list[tuple[str, str, str]] = []
    dataset_calls: list[tuple[str, str, bool]] = []

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
    ) -> str:
        hub_calls.append((repo_id, filename, repo_type))
        return downloaded_script_path

    def fake_load_dataset(
        dataset_path: str,
        builder_config: str,
        *,
        trust_remote_code: bool,
    ) -> object:
        dataset_calls.append((dataset_path, builder_config, trust_remote_code))
        return sentinel_dataset

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = load_official_plantvillage_dataset()

    assert dataset is sentinel_dataset
    assert DATASET_CONFIG == "color"
    assert hub_calls == [(DATASET_ID, HUGGINGFACE_SCRIPT_FILENAME, "dataset")]
    assert dataset_calls == [
        (downloaded_script_path, HUGGINGFACE_SCRIPT_BUILDER_CONFIG, True)
    ]


@pytest.mark.parametrize(
    ("missing_field", "error_match"),
    [
        ("image", "image"),
        ("crop", "crop"),
        ("disease", "disease"),
        ("leaf_id", "leaf_id"),
    ],
)
def test_normalize_plantvillage_records_requires_official_schema_fields(
    missing_field: str,
    error_match: str,
) -> None:
    row: dict[str, object] = {
        "image": Image.new("RGB", (4, 4)),
        "label": 0,
        "crop": "Tomato",
        "disease": "healthy",
        "leaf_id": "leaf a",
    }
    del row[missing_field]
    dataset = {
        "train": _FakeSplit([row], ["Tomato___healthy"]),
        "test": _FakeSplit([], ["Tomato___healthy"]),
    }

    with pytest.raises(PlantVillageError, match=error_match):
        normalize_plantvillage_records(dataset)


def test_normalize_plantvillage_records_rejects_text_only_rows() -> None:
    dataset = {
        "train": _FakeSplit([{"text": "color_train/Tomato___healthy/example.jpg"}], []),
        "test": _FakeSplit([], []),
    }

    with pytest.raises(PlantVillageError, match="class metadata"):
        normalize_plantvillage_records(dataset)


def test_export_filters_tomato_writes_rgb_jpegs_and_source_manifest(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "raw" / "plantvillage_tomato_color"
    manifest_path = tmp_path / "processed" / "plantvillage_source_manifest.csv"
    dataset = _fake_dataset()

    summary = export_tomato_subset(
        output_dir,
        manifest_path,
        dataset=dataset,
        expected_classes=EXPECTED_CLASSES,
    )

    rows = _read_csv(manifest_path)
    relative_paths = [row["relative_path"] for row in rows]
    assert summary.total_records_seen == 4
    assert summary.tomato_records_exported == 3
    assert summary.class_names == EXPECTED_CLASSES
    assert summary.source_split_counts == (("train", 2), ("test", 1))
    assert "Apple___healthy" not in {row["class_name"] for row in rows}
    assert relative_paths == [
        "Tomato___Bacterial_spot/train_leaf_b_000001.jpg",
        "Tomato___healthy/train_leaf_a_000000.jpg",
        "Tomato___healthy/test_leaf_c_000000.jpg",
    ]
    assert all(not Path(row["relative_path"]).is_absolute() for row in rows)
    assert rows[0]["source_split"] == "train"
    assert rows[0]["leaf_id"] == "leaf b"

    exported_path = output_dir / "Tomato___healthy" / "train_leaf_a_000000.jpg"
    assert exported_path.exists()
    with Image.open(exported_path) as exported_image:
        assert exported_image.mode == "RGB"
        assert exported_image.format == "JPEG"


def test_export_protects_existing_images_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "raw"
    manifest_path = tmp_path / "source.csv"
    existing_image = output_dir / "Tomato___healthy" / "train_leaf_a_000000.jpg"
    existing_image.parent.mkdir(parents=True)
    existing_image.write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError, match="existing images"):
        export_tomato_subset(
            output_dir,
            manifest_path,
            dataset=_fake_dataset(),
            expected_classes=EXPECTED_CLASSES,
        )

    output_dir = tmp_path / "fresh_raw"
    manifest_path.write_text("already here", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Source manifest already exists"):
        export_tomato_subset(
            output_dir,
            manifest_path,
            dataset=_fake_dataset(),
            expected_classes=EXPECTED_CLASSES,
        )


def test_export_rejects_missing_metadata(tmp_path: Path) -> None:
    dataset = {
        "train": _FakeSplit(
            [
                {
                    "image": Image.new("RGB", (4, 4)),
                    "crop": "Tomato",
                    "disease": "healthy",
                }
            ],
            ["Tomato___healthy"],
        ),
        "test": _FakeSplit([], ["Tomato___healthy"]),
    }

    with pytest.raises(PlantVillageError, match="leaf_id"):
        export_tomato_subset(
            tmp_path / "raw",
            tmp_path / "source.csv",
            dataset=dataset,
            expected_classes=("Tomato___healthy",),
        )


def test_leaf_group_split_preserves_official_test_and_leaf_groups() -> None:
    source_records = _source_records()

    first = create_leaf_group_manifest(
        source_records,
        validation_ratio=0.34,
        random_seed=7,
        expected_classes=EXPECTED_CLASSES,
    )
    second = create_leaf_group_manifest(
        source_records,
        validation_ratio=0.34,
        random_seed=7,
        expected_classes=EXPECTED_CLASSES,
    )

    assert first == second
    assert len(first) == len(source_records)
    assert {
        record.relative_path.as_posix()
        for record in first
        if record.source_split == "test"
    } == {"Tomato___healthy/test_h4.jpg", "Tomato___Bacterial_spot/test_b4.jpg"}
    assert all(
        record.split == "test" for record in first if record.source_split == "test"
    )
    assert all(
        record.split in {"train", "validation"}
        for record in first
        if record.source_split == "train"
    )
    assert _splits_by_leaf_id(first)["h1"] == {"train"}
    assert {
        (record.class_name, record.split)
        for record in first
        if record.source_split == "train"
    } >= {
        ("Tomato___healthy", "train"),
        ("Tomato___healthy", "validation"),
        ("Tomato___Bacterial_spot", "train"),
        ("Tomato___Bacterial_spot", "validation"),
    }


def test_leaf_group_split_uses_official_test_precedence_for_shared_leaf_ids() -> None:
    source_records = _source_records_with_official_test_conflict()

    first = create_leaf_group_manifest(
        source_records,
        validation_ratio=0.34,
        random_seed=7,
        expected_classes=EXPECTED_CLASSES,
    )
    second = create_leaf_group_manifest(
        source_records,
        validation_ratio=0.34,
        random_seed=7,
        expected_classes=EXPECTED_CLASSES,
    )

    records_by_path = {record.relative_path.as_posix(): record for record in first}
    source_paths = {record.relative_path.as_posix() for record in source_records}
    conflict_records = tuple(
        record for record in first if record.leaf_id == "shared_leaf"
    )
    unrelated_train_splits = {
        record.split
        for record in first
        if record.source_split == "train" and record.leaf_id != "shared_leaf"
    }

    assert first == second
    assert len(first) == len(source_records)
    assert set(records_by_path) == source_paths
    assert len(conflict_records) == 2
    assert {record.source_split for record in conflict_records} == {"train", "test"}
    assert all(record.split == "test" for record in conflict_records)
    assert records_by_path[
        "Tomato___healthy/train_shared_leaf.jpg"
    ].source_split == "train"
    assert records_by_path["Tomato___healthy/train_shared_leaf.jpg"].split == "test"
    assert all(
        record.split == "test" for record in first if record.source_split == "test"
    )
    assert all(len(splits) == 1 for splits in _splits_by_leaf_id(first).values())
    assert unrelated_train_splits == {"train", "validation"}


def test_validate_leaf_group_manifest_rejects_unjustified_train_to_test() -> None:
    source_records = (
        SourceManifestRecord(
            Path("Tomato___healthy/train_h1.jpg"),
            "Tomato___healthy",
            "train",
            "h1",
        ),
        SourceManifestRecord(
            Path("Tomato___healthy/test_h2.jpg"),
            "Tomato___healthy",
            "test",
            "h2",
        ),
    )
    records = (
        FoliaScanManifestRecord(
            Path("Tomato___healthy/train_h1.jpg"),
            "Tomato___healthy",
            "test",
            "h1",
            "train",
        ),
        FoliaScanManifestRecord(
            Path("Tomato___healthy/test_h2.jpg"),
            "Tomato___healthy",
            "test",
            "h2",
            "test",
        ),
    )

    with pytest.raises(PlantVillageError, match="leaf_id also appears"):
        validate_leaf_group_manifest(
            records,
            source_records,
            expected_classes=("Tomato___healthy",),
        )


def test_leaf_group_split_rejects_too_few_leaf_groups() -> None:
    records = (
        SourceManifestRecord(
            Path("Tomato___healthy/train_h1.jpg"),
            "Tomato___healthy",
            "train",
            "h1",
        ),
        SourceManifestRecord(
            Path("Tomato___healthy/test_h2.jpg"),
            "Tomato___healthy",
            "test",
            "h2",
        ),
    )

    with pytest.raises(PlantVillageError, match="at least 2"):
        create_leaf_group_manifest(
            records,
            validation_ratio=0.15,
            random_seed=42,
            expected_classes=("Tomato___healthy",),
        )


def test_source_manifest_validation_rejects_duplicates_and_absolute_paths(
    tmp_path: Path,
) -> None:
    duplicate_manifest = tmp_path / "duplicate.csv"
    duplicate_manifest.write_text(
        "\n".join(
            [
                "relative_path,class_name,source_split,leaf_id",
                "a.jpg,Tomato___healthy,train,h1",
                "a.jpg,Tomato___healthy,train,h2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PlantVillageError, match="Duplicate"):
        read_source_manifest(duplicate_manifest)

    absolute_manifest = tmp_path / "absolute.csv"
    absolute_manifest.write_text(
        "\n".join(
            [
                "relative_path,class_name,source_split,leaf_id",
                f"{tmp_path / 'a.jpg'},Tomato___healthy,train,h1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PlantVillageError, match="relative"):
        read_source_manifest(absolute_manifest)


def test_write_foliascan_manifest_protects_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset_manifest.csv"
    output_path.write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_foliascan_manifest(
            (
                FoliaScanManifestRecord(
                    Path("Tomato___healthy/train_h1.jpg"),
                    "Tomato___healthy",
                    "train",
                    "h1",
                    "train",
                ),
            ),
            output_path,
        )


def test_cli_plantvillage_export_is_mockable_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_export_tomato_subset(
        output_dir: Path,
        source_manifest_path: Path,
        *,
        overwrite: bool = False,
        dataset: Any | None = None,
        expected_classes: tuple[str, ...] = EXPECTED_CLASSES,
    ) -> PlantVillageExportSummary:
        assert overwrite is True
        assert dataset is None
        assert expected_classes == EXPECTED_CLASSES
        return PlantVillageExportSummary(
            dataset_id=DATASET_ID,
            dataset_config=DATASET_CONFIG,
            output_dir=output_dir,
            source_manifest_path=source_manifest_path,
            total_records_seen=3,
            tomato_records_exported=3,
            class_names=EXPECTED_CLASSES,
            source_split_counts=(("train", 2), ("test", 1)),
            image_format="JPEG",
        )

    monkeypatch.setattr(cli, "export_tomato_subset", fake_export_tomato_subset)

    exit_status = cli.main(
        [
            "plantvillage-export",
            "--output-dir",
            str(tmp_path / "raw"),
            "--source-manifest",
            str(tmp_path / "source.csv"),
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    assert exit_status == 0
    assert "multiple gigabytes" in captured.out
    assert "Tomato records exported: 3" in captured.out


def test_cli_plantvillage_split_is_mockable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_create_and_write_leaf_group_manifest(
        source_manifest_path: Path,
        output_path: Path,
        *,
        validation_ratio: float,
        random_seed: int,
        overwrite: bool = False,
        expected_classes: tuple[str, ...] = EXPECTED_CLASSES,
    ) -> tuple[FoliaScanManifestRecord, ...]:
        assert source_manifest_path == tmp_path / "source.csv"
        assert output_path == tmp_path / "dataset_manifest.csv"
        assert validation_ratio == 0.2
        assert random_seed == 99
        assert overwrite is True
        assert expected_classes == EXPECTED_CLASSES
        return (
            FoliaScanManifestRecord(
                Path("Tomato___healthy/train_h1.jpg"),
                "Tomato___healthy",
                "train",
                "h1",
                "train",
            ),
            FoliaScanManifestRecord(
                Path("Tomato___healthy/test_h2.jpg"),
                "Tomato___healthy",
                "test",
                "h2",
                "test",
            ),
        )

    monkeypatch.setattr(
        cli,
        "create_and_write_leaf_group_manifest",
        fake_create_and_write_leaf_group_manifest,
    )

    exit_status = cli.main(
        [
            "plantvillage-split",
            "--source-manifest",
            str(tmp_path / "source.csv"),
            "--output",
            str(tmp_path / "dataset_manifest.csv"),
            "--validation-ratio",
            "0.2",
            "--random-seed",
            "99",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    assert exit_status == 0
    assert "PlantVillage leaf-group split complete" in captured.out
    assert "train: 1" in captured.out
    assert "test: 1" in captured.out


class _FakeLabelFeature:
    def __init__(self, names: list[str]) -> None:
        self.names = names


class _FakeSplit(list[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]], label_names: list[str]) -> None:
        super().__init__(rows)
        self.features = {"label": _FakeLabelFeature(label_names)}


def _fake_dataset() -> dict[str, _FakeSplit]:
    label_names = [
        "Apple___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___healthy",
    ]
    return {
        "train": _FakeSplit(
            [
                {
                    "image": Image.new("L", (4, 4)),
                    "label": 2,
                    "crop": "Tomato",
                    "disease": "healthy",
                    "leaf_id": "leaf a",
                },
                {
                    "image": Image.new("RGB", (4, 4)),
                    "label": 1,
                    "crop": "Tomato",
                    "disease": "Bacterial_spot",
                    "leaf_id": "leaf b",
                },
                {
                    "image": Image.new("RGB", (4, 4)),
                    "label": 0,
                    "crop": "Apple",
                    "disease": "healthy",
                    "leaf_id": "apple leaf",
                },
            ],
            label_names,
        ),
        "test": _FakeSplit(
            [
                {
                    "image": Image.new("RGB", (4, 4)),
                    "label": 2,
                    "crop": "Tomato",
                    "disease": "healthy",
                    "leaf_id": "leaf c",
                },
            ],
            label_names,
        ),
    }


def _source_records() -> tuple[SourceManifestRecord, ...]:
    rows = [
        ("Tomato___healthy/train_h1_a.jpg", "Tomato___healthy", "train", "h1"),
        ("Tomato___healthy/train_h1_b.jpg", "Tomato___healthy", "train", "h1"),
        ("Tomato___healthy/train_h2.jpg", "Tomato___healthy", "train", "h2"),
        ("Tomato___healthy/train_h3.jpg", "Tomato___healthy", "train", "h3"),
        ("Tomato___healthy/test_h4.jpg", "Tomato___healthy", "test", "h4"),
        (
            "Tomato___Bacterial_spot/train_b1.jpg",
            "Tomato___Bacterial_spot",
            "train",
            "b1",
        ),
        (
            "Tomato___Bacterial_spot/train_b2.jpg",
            "Tomato___Bacterial_spot",
            "train",
            "b2",
        ),
        (
            "Tomato___Bacterial_spot/train_b3.jpg",
            "Tomato___Bacterial_spot",
            "train",
            "b3",
        ),
        (
            "Tomato___Bacterial_spot/test_b4.jpg",
            "Tomato___Bacterial_spot",
            "test",
            "b4",
        ),
    ]
    return tuple(
        SourceManifestRecord(Path(path), class_name, source_split, leaf_id)
        for path, class_name, source_split, leaf_id in rows
    )


def _source_records_with_official_test_conflict() -> tuple[SourceManifestRecord, ...]:
    rows = [
        (
            "Tomato___healthy/train_shared_leaf.jpg",
            "Tomato___healthy",
            "train",
            "shared_leaf",
        ),
        (
            "Tomato___healthy/test_shared_leaf.jpg",
            "Tomato___healthy",
            "test",
            "shared_leaf",
        ),
        ("Tomato___healthy/train_h1.jpg", "Tomato___healthy", "train", "h1"),
        ("Tomato___healthy/train_h2.jpg", "Tomato___healthy", "train", "h2"),
        ("Tomato___healthy/train_h3.jpg", "Tomato___healthy", "train", "h3"),
        ("Tomato___healthy/test_h4.jpg", "Tomato___healthy", "test", "h4"),
        (
            "Tomato___Bacterial_spot/train_b1.jpg",
            "Tomato___Bacterial_spot",
            "train",
            "b1",
        ),
        (
            "Tomato___Bacterial_spot/train_b2.jpg",
            "Tomato___Bacterial_spot",
            "train",
            "b2",
        ),
        (
            "Tomato___Bacterial_spot/train_b3.jpg",
            "Tomato___Bacterial_spot",
            "train",
            "b3",
        ),
        (
            "Tomato___Bacterial_spot/test_b4.jpg",
            "Tomato___Bacterial_spot",
            "test",
            "b4",
        ),
    ]
    return tuple(
        SourceManifestRecord(Path(path), class_name, source_split, leaf_id)
        for path, class_name, source_split, leaf_id in rows
    )


def _splits_by_leaf_id(
    records: tuple[FoliaScanManifestRecord, ...],
) -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {}
    for record in records:
        splits.setdefault(record.leaf_id, set()).add(record.split)
    return splits


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))
