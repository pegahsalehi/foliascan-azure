# Local Training Foundation

Phase 2A adds the local data and model foundation for a simple image
classification baseline. It does not train a model, save weights, calculate
final accuracy, or touch Azure services.

## Baseline Model

FoliaScan starts with ResNet18 because it is a well-known, compact convolutional
baseline with predictable torchvision support. It is strong enough to validate
the data path and model shape without adding architecture complexity too early.

Transfer learning is supported through the `pretrained` setting. When
`freeze_backbone` is enabled, FoliaScan freezes the ResNet feature extractor and
keeps the final classification layer trainable for the project-specific tomato
classes.

## Manifest-Driven Loading

Training data is read from the leakage-safe FoliaScan manifest:

```text
relative_path,class_name,split,leaf_id,source_split
```

The loader preserves `relative_path` as a `Path`, rejects absolute or duplicate
paths, validates split names, and loads images lazily with Pillow. Images are
converted to RGB before tensor conversion.

Class indexes are derived from the manifest by sorting class names
deterministically. The same mapping is used for train, validation, and test, and
numeric PlantVillage label IDs are not hard-coded.

## Transforms

Training transforms resize images, apply a conservative random crop, horizontal
flip, and small rotation, then use ImageNet normalization. The augmentation is
intentionally restrained because leaf orientation and disease texture can carry
important signal.

Validation and test transforms are deterministic: resize, center crop, tensor
conversion, and the same ImageNet normalization.

## Split Integrity

The leakage-safe splits created in Phase 1B2 must remain unchanged. The training
pipeline filters records by the manifest `split` column and does not resplit
images. The original `source_split` and `leaf_id` fields remain available for
auditability.

## Smoke Test

Run one forward-pass smoke test after the local PlantVillage export and
leakage-safe manifest exist:

```powershell
poetry run python -m foliascan.training.smoke_test `
  --manifest data/processed/dataset_manifest.csv `
  --data-dir data/raw/plantvillage_tomato_color `
  --config configs/training.example.yaml `
  --split train
```

The smoke test loads one batch, builds ResNet18, runs a single forward pass, and
checks that the output shape is `batch_size x number_of_classes`. It uses CPU by
default unless an available device is explicitly requested.

Full model training is intentionally left for the next phase.
