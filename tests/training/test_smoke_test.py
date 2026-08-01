import csv
from pathlib import Path

from PIL import Image

from foliascan.training.dataset import MANIFEST_COLUMNS
from foliascan.training.smoke_test import SmokeTestSummary, main, run_smoke_test


def test_run_smoke_test_performs_successful_forward_pass(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    manifest_path = tmp_path / "dataset_manifest.csv"
    config_path = tmp_path / "training.yaml"
    _write_training_files(data_dir, manifest_path, config_path)

    summary = run_smoke_test(
        manifest_path=manifest_path,
        data_dir=data_dir,
        config_path=config_path,
        split="train",
    )

    assert summary.split == "train"
    assert summary.batch_shape == (2, 3, 32, 32)
    assert summary.target_shape == (2,)
    assert summary.num_classes == 2
    assert summary.output_shape == (2, 2)
    assert summary.device == "cpu"


def test_smoke_test_cli_parses_arguments_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    def fake_run_smoke_test(
        *,
        manifest_path: Path,
        data_dir: Path,
        config_path: Path,
        split: str,
        device_name: str = "cpu",
    ) -> SmokeTestSummary:
        assert manifest_path == tmp_path / "dataset_manifest.csv"
        assert data_dir == tmp_path / "raw"
        assert config_path == tmp_path / "training.yaml"
        assert split == "validation"
        assert device_name == "cpu"
        return SmokeTestSummary(
            split="validation",
            batch_shape=(1, 3, 32, 32),
            target_shape=(1,),
            num_classes=2,
            output_shape=(1, 2),
            device="cpu",
        )

    monkeypatch.setattr(
        "foliascan.training.smoke_test.run_smoke_test",
        fake_run_smoke_test,
    )

    exit_status = main(
        [
            "--manifest",
            str(tmp_path / "dataset_manifest.csv"),
            "--data-dir",
            str(tmp_path / "raw"),
            "--config",
            str(tmp_path / "training.yaml"),
            "--split",
            "validation",
        ]
    )

    captured = capsys.readouterr()
    assert exit_status == 0
    assert "FoliaScan training smoke test complete" in captured.out
    assert "split: validation" in captured.out


def _write_training_files(
    data_dir: Path,
    manifest_path: Path,
    config_path: Path,
) -> None:
    rows = [
        ("Tomato___healthy/train_0.jpg", "Tomato___healthy", "train", "h1", "train"),
        (
            "Tomato___Bacterial_spot/train_0.jpg",
            "Tomato___Bacterial_spot",
            "train",
            "b1",
            "train",
        ),
        (
            "Tomato___healthy/validation_0.jpg",
            "Tomato___healthy",
            "validation",
            "h2",
            "train",
        ),
        (
            "Tomato___Bacterial_spot/test_0.jpg",
            "Tomato___Bacterial_spot",
            "test",
            "b2",
            "test",
        ),
    ]
    for relative_path, _, _, _, _ in rows:
        image_path = data_dir / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), color=(20, 120, 70)).save(image_path)

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for relative_path, class_name, split, leaf_id, source_split in rows:
            writer.writerow(
                {
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "split": split,
                    "leaf_id": leaf_id,
                    "source_split": source_split,
                }
            )

    config_path.write_text(
        "\n".join(
            [
                "random_seed: 42",
                "image_size: 32",
                "batch_size: 2",
                "learning_rate: 0.001",
                "epochs: 1",
                "model_name: resnet18",
                "num_workers: 0",
                "pretrained: false",
                "freeze_backbone: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
