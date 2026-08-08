# Dataset Preparation

FoliaScan uses the PlantVillage Tomato dataset for multi-class tomato-leaf image classification.

The dataset preparation workflow exports the tomato subset, preserves source metadata, and creates leakage-safe train, validation, and test splits.

The raw dataset is not committed or redistributed with this repository.

## PlantVillage Source

The dataset is loaded from the Hugging Face dataset:

```text
mohanty/PlantVillage
```

using the:

```text
color
```

configuration.

The export process filters records to:

```text
crop == "Tomato"
```

and writes only the tomato subset locally.

The expected classes are:

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

Class names are validated from dataset metadata rather than relying on hard-coded numeric label IDs.

Source records preserve metadata including:

- image
- crop
- disease
- class information
- `leaf_id`
- original source split

## Leakage-Safe Splitting

PlantVillage provides a `leaf_id` that identifies images originating from the same physical leaf.

FoliaScan uses this identifier to keep related images in the same final split.

This is important because placing different views or augmentations of the same leaf across training and evaluation splits could cause data leakage and produce overly optimistic metrics.

Official PlantVillage test records remain in the final test split.

A small number of fallback `leaf_id` values can appear in both the original train and test data. FoliaScan applies test precedence to those groups: if a `leaf_id` appears in the official test split, all records with that identifier are assigned to the final test split.

The original source split is still preserved for auditability.

For example:

```text
source_split=train
split=test
```

indicates that the record originally belonged to the source training data but was moved to the final test split to keep its leaf group together.

Remaining training leaf groups are divided between the FoliaScan training and validation splits.

The PlantVillage workflow therefore splits at `leaf_id` level rather than individual image level.

## Export the Tomato Dataset

Run:

```powershell
poetry run python -m foliascan.data.cli plantvillage-export `
  --output-dir data/raw/plantvillage_tomato_color `
  --source-manifest data/processed/plantvillage_source_manifest.csv
```

The export downloads the source dataset and writes the tomato subset as RGB JPEG images.

Source metadata is stored in:

```text
data/processed/plantvillage_source_manifest.csv
```

The source manifest contains:

- `relative_path`
- `class_name`
- `source_split`
- `leaf_id`

Use `--overwrite` only when intentionally replacing an existing export.

## Create the FoliaScan Manifest

After export, create the leakage-safe project manifest:

```powershell
poetry run python -m foliascan.data.cli plantvillage-split `
  --source-manifest data/processed/plantvillage_source_manifest.csv `
  --output data/processed/dataset_manifest.csv `
  --validation-ratio 0.15 `
  --random-seed 42
```

The resulting manifest contains:

```text
relative_path
class_name
split
leaf_id
source_split
```

Possible final split values are:

```text
train
validation
test
```

The manifest defines the dataset split used throughout training and evaluation. Downstream pipelines do not resplit the images.

## Generic Folder-Based Workflow

FoliaScan also includes generic utilities for simple image datasets organized as one directory per class:

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

Supported image formats are:

```text
.jpg
.jpeg
.png
```

Inspect a folder-based dataset with:

```powershell
poetry run python -m foliascan.data.cli inspect `
  --data-dir data/raw/plantvillage_tomato_color
```

Optionally write a JSON report:

```powershell
poetry run python -m foliascan.data.cli inspect `
  --data-dir data/raw/plantvillage_tomato_color `
  --json-report data/processed/dataset_report.json
```

Create a generic stratified manifest with:

```powershell
poetry run python -m foliascan.data.cli split `
  --data-dir data/raw/plantvillage_tomato_color `
  --output data/processed/dataset_manifest.csv
```

The generic workflow splits individual images and does not understand PlantVillage `leaf_id` groups. It should therefore not replace the PlantVillage-specific split workflow for this project.

## Dataset Limitations

PlantVillage contains largely controlled images, so its visual distribution can differ substantially from real field conditions.

A model trained on this dataset may be affected by:

- background bias
- controlled lighting
- framing differences
- camera differences
- weather conditions
- leaf occlusion
- plant variety
- mixed disease states
- other forms of domain shift

Performance on PlantVillage should therefore not be interpreted as equivalent to real-world diagnostic performance.

FoliaScan is not a professional agricultural diagnostic tool and should not replace assessment by qualified agricultural specialists.

## License and Citation

The source dataset documentation lists the dataset under CC BY-SA 3.0. Licensing and redistribution requirements should be reviewed before sharing source images or derived dataset artifacts.

Citation:

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

Raw and generated dataset files under:

```text
data/raw/
data/processed/
```

are excluded from Git by default.