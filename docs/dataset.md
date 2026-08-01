# Dataset Preparation

Phase 1B prepares local image datasets for the initial tomato-leaf
classification scope. Phase 1B2 adds official PlantVillage ingestion and
leakage-safe split preparation. This repository does not download, commit, or
redistribute the dataset.

## Official PlantVillage Source

Use the Hugging Face dataset `mohanty/PlantVillage` with the `color`
configuration. The export command filters the dataset to `crop == "Tomato"` and
writes only the tomato subset locally.

Expected tomato condition classes:

- `Tomato___Bacterial_spot`
- `Tomato___Early_blight`
- `Tomato___Late_blight`
- `Tomato___Leaf_Mold`
- `Tomato___Septoria_leaf_spot`
- `Tomato___Spider_mites Two-spotted_spider_mite`
- `Tomato___Target_Spot`
- `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
- `Tomato___Tomato_mosaic_virus`
- `Tomato___healthy`

The official Hugging Face records are expected to include `image`, class
metadata, `crop`, `disease`, `leaf_id`, and the official source split. The
implementation validates class names from dataset metadata instead of relying on
hard-coded numeric label IDs.

## Leaf-Group Splitting

PlantVillage provides a `leaf_id` so images from the same physical leaf can stay
together. FoliaScan preserves every official PlantVillage `test` record as
`test`. Real upstream data can contain a small number of fallback leaf IDs that
appear in both official `train` and official `test`. FoliaScan applies test
precedence to those groups: every record sharing an official-test `leaf_id` is
assigned to final `test`.

This may move a small number of official-training records into the final
FoliaScan `test` split. The original `source_split` value is retained, so those
records remain auditable as `source_split=train` and `split=test`. Official
training leaf IDs that are absent from official `test` are divided into
FoliaScan `train` and `validation` subsets.

This prevents leakage: if augmented or related views of the same leaf appeared
in different splits, validation or test metrics could overstate model quality.
The PlantVillage-specific split command therefore operates at `leaf_id` group
level. The generic folder-based split command is not used for PlantVillage
because it splits individual images and does not know about `leaf_id`.

## Export PlantVillage Tomato

The export command downloads the official dataset through Hugging Face. Expect a
large, multi-gigabyte download. Run it manually only when you are ready to store
the raw data locally:

```powershell
poetry run python -m foliascan.data.cli plantvillage-export `
  --output-dir data/raw/plantvillage_tomato_color `
  --source-manifest data/processed/plantvillage_source_manifest.csv
```

The exporter writes RGB JPEG files at quality 95 for consistency across source
image modes and preserves the source metadata in:

```text
data/processed/plantvillage_source_manifest.csv
```

Source manifest columns:

- `relative_path`: image path relative to `data/raw/plantvillage_tomato_color`
- `class_name`: tomato condition class
- `source_split`: official PlantVillage `train` or `test`
- `leaf_id`: physical leaf group identifier

Use `--overwrite` only when you intentionally want to replace existing exported
images and the source manifest.

## Create The FoliaScan Manifest

After export, create the leakage-safe project manifest:

```powershell
poetry run python -m foliascan.data.cli plantvillage-split `
  --source-manifest data/processed/plantvillage_source_manifest.csv `
  --output data/processed/dataset_manifest.csv `
  --validation-ratio 0.15 `
  --random-seed 42
```

The FoliaScan manifest contains:

- `relative_path`: image path relative to the exported dataset root
- `class_name`: tomato condition class
- `split`: FoliaScan `train`, `validation`, or `test`
- `leaf_id`: physical leaf group identifier
- `source_split`: official PlantVillage `train` or `test`

Use `--overwrite` only when replacing an existing manifest is intentional.

## Generic Folder-Based Workflow

The generic tools remain available for simple local datasets with one immediate
subdirectory per class:

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

Inspect a local directory-based dataset:

```powershell
poetry run python -m foliascan.data.cli inspect --data-dir data/raw/plantvillage_tomato_color
```

To also write a JSON report:

```powershell
poetry run python -m foliascan.data.cli inspect --data-dir data/raw/plantvillage_tomato_color --json-report data/processed/dataset_report.json
```

Generate a generic stratified manifest from valid images:

```powershell
poetry run python -m foliascan.data.cli split --data-dir data/raw/plantvillage_tomato_color --output data/processed/dataset_manifest.csv
```

The generic manifest contains only `relative_path`, `class_name`, and `split`.
It uses only valid images, does not copy or move image files, and requires
`--overwrite` before replacing an existing manifest.

## Limitations

Controlled image datasets can contain background, lighting, framing, and capture
conditions that are very different from field images. A model trained only on
controlled images may learn background bias or other shortcuts instead of robust
leaf-condition features. Field images can introduce domain shift from different
cameras, weather, soil, plant varieties, leaf occlusion, and mixed disease
states.

FoliaScan is educational software. It must not be treated as a diagnostic tool,
and any future predictions must not replace advice from agricultural
specialists, agronomists, plant pathologists, or other qualified professionals.

## License And Citation

The Hugging Face dataset card currently lists `cc-by-sa-3.0`, and the upstream
loader describes the license as CC BY-SA 3.0 subject to verification. Verify the
license, permitted uses, attribution, share-alike obligations, and redistribution
rules before sharing the dataset or derived image artifacts.

Required citation:

```bibtex
@article{Mohanty_Hughes_Salathe_2016,
  title = {Using deep learning for image-based plant disease detection},
  volume = {7},
  doi = {10.3389/fpls.2016.01419},
  journal = {Frontiers in Plant Science},
  author = {Mohanty, Sharada P. and Hughes, David P. and Salathe, Marcel},
  year = {2016},
  month = {Sep}
}
```

Raw datasets and generated data under `data/raw/` and `data/processed/` are
excluded from Git by default.
