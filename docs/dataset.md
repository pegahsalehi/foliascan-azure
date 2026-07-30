# Dataset Preparation

Phase 1B prepares local image datasets for the initial tomato-leaf
classification scope. The intended future dataset is the tomato subset of the
PlantVillage color-image dataset, but this repository does not download or
redistribute that dataset.

## Expected Structure

Use one immediate subdirectory per tomato-leaf condition class:

```text
dataset_root/
|-- Tomato___Bacterial_spot/
|   |-- image_001.jpg
|   `-- image_002.jpg
|-- Tomato___Early_blight/
|   `-- image_003.jpg
`-- Tomato___healthy/
    `-- image_004.jpg
```

Supported image extensions are `.jpg`, `.jpeg`, and `.png`, matched
case-insensitively. Hidden files, hidden directories, unsupported files, and
nested directories below a class directory are ignored.

## Why Folder-Based Data

A public folder-based image dataset is appropriate for the first local version
because it is simple to inspect, easy to split reproducibly, and does not require
cloud storage or a database before the data workflow is validated. It also keeps
the class label convention transparent: each immediate directory name is the
class label.

## Limitations

Controlled image datasets can contain background, lighting, framing, and capture
conditions that are very different from field images. A model trained only on
controlled images may learn background bias or other shortcuts instead of robust
leaf-condition features. Field images can introduce domain shift from different
cameras, weather, soil, plant varieties, leaf occlusion, and mixed disease
states.

LeafSignal AI is educational software. It must not be treated as a diagnostic
tool, and any future predictions must not replace advice from agricultural
specialists, agronomists, plant pathologists, or other qualified professionals.

Before redistributing any dataset or derived artifact, verify the dataset
license, permitted uses, and required citation.

## Inspect A Dataset

```powershell
poetry run python -m leafsignal.data.cli inspect --data-dir data/raw/plantvillage_tomato_color
```

To also write a JSON report:

```powershell
poetry run python -m leafsignal.data.cli inspect --data-dir data/raw/plantvillage_tomato_color --json-report data/processed/dataset_report.json
```

Inspection discovers supported files, validates them with Pillow, counts classes
and extensions, records image modes and sizes, and reports corrupted images
without modifying source files.

## Generate A Manifest

```powershell
poetry run python -m leafsignal.data.cli split --data-dir data/raw/plantvillage_tomato_color --output data/processed/dataset_manifest.csv
```

The manifest contains:

- `relative_path`: image path relative to the dataset root.
- `class_name`: class label from the immediate parent directory.
- `split`: one of `train`, `validation`, or `test`.

The split command uses only valid images, does not copy or move image files, and
requires `--overwrite` before replacing an existing manifest.

Raw datasets and generated data under `data/raw/` and `data/processed/` are
excluded from Git by default.

