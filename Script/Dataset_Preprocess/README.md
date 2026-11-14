# MIPDB Dataset Preprocessing

These scripts belong to the same MIPDB project, but they are **alternative
conversion branches**, not a sequential pipeline.

## Directory structure

```text
Dataset_Preprocess/
├── commands/                  # Current task-specific entry points
│   ├── convert_yolo_pose.py
│   ├── tile_uav_pose.py
│   ├── convert_plant_detection.py
│   ├── convert_coco.py
│   ├── process_points.py
│   └── point_resampling.py
├── legacy/                    # Historical combined workflow (not required)
│   └── data_preprocess.py
└── README.md
```

## Choose a command by the required output

| Input | Output | Use |
|---|---|---|
| Labelme JSON | YOLO pose (`stem` and `leaf`) for DSLR or ordinary images | [`commands/convert_yolo_pose.py`](./commands/convert_yolo_pose.py) |
| Labelme JSON + large UAV images | tiled images and YOLO pose labels | [`commands/tile_uav_pose.py`](./commands/tile_uav_pose.py) |
| Labelme JSON | one YOLO detection box per plant | [`commands/convert_plant_detection.py`](./commands/convert_plant_detection.py) |
| Labelme JSON | COCO keypoint JSON | [`commands/convert_coco.py`](./commands/convert_coco.py) |
| Labelme JSON | fixed-count resampled Labelme JSON | [`commands/process_points.py`](./commands/process_points.py) |

The current commands are independent entry points. They do not call
`legacy/data_preprocess.py`, and `legacy/data_preprocess.py` is not a
prerequisite for any command.

```text
Labelme JSON
├── ordinary YOLO pose     -> commands/convert_yolo_pose.py
├── tiled UAV YOLO pose    -> commands/tile_uav_pose.py
├── plant detection        -> commands/convert_plant_detection.py
└── COCO keypoints         -> commands/convert_coco.py
```

## Important format differences

- `stem` and `leaf` are separate classes in the pose and COCO branches.
- The plant-detection branch groups labels sharing a plant ID and emits one
  `plant` bounding box.
- The UAV branch resamples linestrips to the requested keypoint count and also
  performs tiling.
- [`legacy/data_preprocess.py`](./legacy/data_preprocess.py) combines image
  extraction, YOLO conversion, and point processing in one command. Use it
  only when reproducing that older workflow.

## Coordinate and keypoint settings

The current conversion branches expose their settings as command-line
arguments. Check the command help before conversion:

```powershell
python .\commands\convert_yolo_pose.py --help
python .\commands\tile_uav_pose.py --help
python .\commands\convert_plant_detection.py --help
python .\commands\convert_coco.py --help
```

Do not mix outputs produced by different branches without recording the
branch, resize factor, bounding-box expansion, and keypoint count. These
parameters affect coordinates and evaluation results.

The legacy point-processing entry point supports two resampling modes:

```powershell
python .\commands\process_points.py `
  --json_dir <input-json> --output_dir <output-json> `
  --num_points 17 --interpolation_method arc_length
```

The same `--interpolation_method {legacy,arc_length}` option is available in
`convert_coco.py`, `convert_yolo_pose.py`, and `tile_uav_pose.py`. For example:

```powershell
python .\commands\convert_coco.py `
  --json_dir <input-json> --output <output-json> `
  --interpolation_method arc_length
```

The historical script remains available only for reproducing its old combined
workflow:

```powershell
python .\legacy\data_preprocess.py --task process `
  --json-dir <input-json> --save-dir <output-json> `
  --num-points 17 --interpolation-method arc_length
```

`legacy` retains the historical height-based root ordering. The current
`arc_length` mode retains input polyline order and resamples at uniform
cumulative arc-length intervals. It requires input points to already follow
the intended anatomical path; it does not correct point-order errors or
reconstruct hidden portions of occluded leaves.

## Relationship to the manuscript

The point-line annotations are the primary MIPDB resource. Detection boxes are
derived from point-line annotations, and the segmentation experiment is
separate and exploratory. This directory contains conversion utilities; it
does not define which plants are foreground annotations. See
[`MIPDB_Markdown/mpdb_markdown.md`](../../MIPDB_Markdown/mpdb_markdown.md) for
the annotation protocol.

## Maintenance policy

New work should use the dedicated task-specific commands. The legacy script is
isolated so that it remains available for reproducing historical outputs
without being mistaken for the current pipeline. Shared Labelme parsing,
label mapping, geometry, and split logic should be extracted only when the
conversion behavior is covered by representative tests.
