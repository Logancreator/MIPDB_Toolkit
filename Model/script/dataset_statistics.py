
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset statistics and visualization for COCO Keypoints and YOLO Pose.

Supported formats:
    1. COCO Keypoints JSON
    2. YOLO Pose TXT labels

Outputs:
    bbox_dimensions.pdf
    keypoint_spatial_distribution.pdf
    bbox_keypoint_count_hist.pdf
    coco_bbox_stats.csv
    coco_keypoint_stats.csv

Examples:
    COCO:
        python coco_stats.py \
            --format coco \
            --coco_json /path/to/annotations.json

    YOLO Pose:
        python coco_stats.py \
            --format yolo \
            --images_dir /path/to/images/train \
            --labels_dir /path/to/labels/train \
            --kpt_dims 3 \
            --output_dir ./yolo_statistics
"""

import os
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image


# ============================================================
# 1. Command-line arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute statistics and visualizations for "
            "COCO Keypoints or YOLO Pose datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--format",
        choices=["coco", "yolo"],
        default="coco",
        help="Annotation format.",
    )

    parser.add_argument(
        "--coco_json",
        type=str,
        default=None,
        help="Path to COCO annotations JSON file.",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="YOLO dataset root containing images/ and labels/ directories.",
    )
    parser.add_argument(
        "--kpt_dims",
        type=int,
        choices=[2, 3],
        default=3,
        help=(
            "Number of values per keypoint in YOLO labels: "
            "2 for x,y; 3 for x,y,visibility. "
            "Ignored for COCO format."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save figures and CSVs. "
            "Defaults to the COCO JSON directory or "
            "./yolo_statistics for YOLO."
        ),
    )

    parser.add_argument(
        "--max_samples_kde",
        type=int,
        default=50000,
        help="Maximum number of samples used for KDE plots.",
    )

    parser.add_argument(
        "--no_bbox_plot",
        action="store_true",
        help="Disable bounding box dimension plots.",
    )

    parser.add_argument(
        "--no_kpt_plot",
        action="store_true",
        help="Disable keypoint spatial distribution plots.",
    )

    return parser.parse_args()


# ============================================================
# 2. Load COCO Keypoints dataset
# ============================================================

def load_coco_dataset(coco_json_path: str) -> dict:
    if not os.path.isfile(coco_json_path):
        raise FileNotFoundError(
            f"COCO annotation file not found: {coco_json_path}"
        )

    print(f"Loading COCO annotations from: {coco_json_path}")

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    for field in ("images", "annotations"):
        if field not in coco:
            raise ValueError(
                f"Invalid COCO JSON: missing '{field}' field."
            )

    print(f"Images in JSON: {len(coco['images'])}")
    print(f"Annotations in JSON: {len(coco['annotations'])}")

    return coco


# ============================================================
# 3. Load YOLO Pose dataset and convert to COCO-like structure
# ============================================================

def load_yolo_pose_dataset(
    images_dir: str,
    labels_dir: str,
    kpt_dims: int = 3,
) -> dict:
    """
    Convert YOLO Pose labels to a COCO-like in-memory structure.

    Expected directory layout:

        dataset/
        ├── images/
        │   ├── image001.jpg
        │   └── subfolder/image002.jpg
        └── labels/
            ├── image001.txt
            └── subfolder/image002.txt

    YOLO label format:

        class cx cy w h x1 y1 v1 x2 y2 v2 ...

    or, when kpt_dims=2:

        class cx cy w h x1 y1 x2 y2 ...

    Bounding boxes and keypoint coordinates are normalized to [0, 1]
    in standard YOLO labels and are converted to pixel coordinates.

    For kpt_dims=2, (0, 0) is treated as an unlabeled keypoint.
    This convention should be checked against the dataset's annotation
    specification before use.
    """

    image_root = Path(images_dir)
    label_root = Path(labels_dir)

    if not image_root.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_root}"
        )

    if not label_root.is_dir():
        raise FileNotFoundError(
            f"Label directory not found: {label_root}"
        )

    if kpt_dims not in (2, 3):
        raise ValueError("--kpt_dims must be either 2 or 3.")

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    image_paths = sorted(
        p for p in image_root.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    if not image_paths:
        raise ValueError(
            f"No images found in directory: {image_root}"
        )

    images = []
    annotations = []

    annotation_id = 0
    missing_label_count = 0
    empty_label_count = 0
    invalid_line_count = 0

    print(f"Loading YOLO images from: {image_root}")
    print(f"Loading YOLO labels from: {label_root}")
    print(f"Keypoint dimensions: {kpt_dims}")

    for image_id, image_path in enumerate(image_paths):
        relative_path = image_path.relative_to(image_root)

        # Match images and labels using their relative paths.
        label_path = (label_root / relative_path).with_suffix(".txt")

        try:
            with Image.open(image_path) as image:
                img_w, img_h = image.size
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read image: {image_path}"
            ) from exc

        if img_w <= 0 or img_h <= 0:
            print(f"Warning: invalid image dimensions: {image_path}")
            continue

        images.append({
            "id": image_id,
            "file_name": str(relative_path),
            "width": img_w,
            "height": img_h,
        })

        # Keep images even when their label file is missing or empty.
        if not label_path.exists():
            missing_label_count += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            empty_label_count += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            try:
                values = [float(value) for value in line.split()]
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in {label_path}, "
                    f"line {line_number}: {line}"
                ) from exc

            if len(values) < 5:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_path}, "
                    f"line {line_number}: expected at least 5 values."
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Non-finite value in {label_path}, "
                    f"line {line_number}."
                )

            class_value = values[0]

            if not class_value.is_integer() or class_value < 0:
                raise ValueError(
                    f"Invalid class ID in {label_path}, "
                    f"line {line_number}: {class_value}"
                )

            class_id = int(class_value)

            # YOLO normalized bbox: center_x, center_y, width, height.
            cx, cy, bw, bh = values[1:5]

            if bw <= 0 or bh <= 0:
                invalid_line_count += 1
                continue

            # Convert bbox to COCO xywh in pixels.
            bbox_x = (cx - bw / 2.0) * img_w
            bbox_y = (cy - bh / 2.0) * img_h
            bbox_w = bw * img_w
            bbox_h = bh * img_h

            keypoint_values = values[5:]

            if keypoint_values and len(keypoint_values) % kpt_dims != 0:
                raise ValueError(
                    f"Invalid keypoint value count in {label_path}, "
                    f"line {line_number}: found "
                    f"{len(keypoint_values)} values after bbox, "
                    f"but --kpt_dims={kpt_dims}."
                )

            keypoints = []

            for i in range(0, len(keypoint_values), kpt_dims):
                kx = keypoint_values[i]
                ky = keypoint_values[i + 1]

                # Standard YOLO keypoint coordinates are normalized.
                pixel_x = kx * img_w
                pixel_y = ky * img_h

                if kpt_dims == 3:
                    visibility_value = keypoint_values[i + 2]

                    if not visibility_value.is_integer():
                        raise ValueError(
                            f"Non-integer visibility value in "
                            f"{label_path}, line {line_number}."
                        )

                    visibility = int(visibility_value)

                    if visibility not in (0, 1, 2):
                        raise ValueError(
                            f"Invalid keypoint visibility "
                            f"{visibility} in {label_path}, "
                            f"line {line_number}; expected 0, 1, or 2."
                        )

                else:
                    # For x,y-only labels, infer missing points using
                    # the common zero-coordinate convention.
                    visibility = (
                        0 if kx == 0 and ky == 0 else 2
                    )

                keypoints.extend([
                    pixel_x,
                    pixel_y,
                    visibility,
                ])

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                ],
                "keypoints": keypoints,
            })

            annotation_id += 1

    print(f"Images loaded: {len(images)}")
    print(f"Annotations loaded: {len(annotations)}")
    print(f"Missing label files: {missing_label_count}")
    print(f"Empty label files: {empty_label_count}")
    print(f"Invalid bbox lines skipped: {invalid_line_count}")

    return {
        "images": images,
        "annotations": annotations,
        "categories": [],
    }


# ============================================================
# 4. Parse annotations into DataFrames
# ============================================================

def build_statistics(coco: dict):
    images_by_id = {
        img["id"]: img for img in coco["images"]
    }

    # Initialize all image counts to zero so images without annotations
    # are included in the per-image histograms.
    bbox_counts = {
        img["id"]: 0 for img in coco["images"]
    }

    kpt_counts = {
        img["id"]: 0 for img in coco["images"]
    }

    bbox_rows = []
    kpt_rows = []

    print("Parsing annotations and building statistics...")

    for ann in coco["annotations"]:
        img_info = images_by_id.get(ann["image_id"])

        if img_info is None:
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        if img_w <= 0 or img_h <= 0:
            continue

        # ----------------------------------------------------
        # Bounding box statistics
        # ----------------------------------------------------
        bbox = ann.get("bbox", [])

        if len(bbox) != 4:
            continue

        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            continue

        bbox_counts[ann["image_id"]] += 1

        bbox_rows.append({
            "image_id": ann["image_id"],
            "category_id": ann.get("category_id", -1),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "area": w * h,
            "aspect_ratio": w / h,
            "img_w": img_w,
            "img_h": img_h,
        })

        # ----------------------------------------------------
        # Keypoint statistics
        # ----------------------------------------------------
        keypoints = ann.get("keypoints", [])

        if not keypoints:
            continue

        if len(keypoints) % 3 != 0:
            print(
                f"Warning: malformed keypoints in annotation "
                f"{ann.get('id', 'unknown')}; skipping keypoints."
            )
            continue

        num_keypoints = len(keypoints) // 3

        for k in range(num_keypoints):
            kx = keypoints[3 * k]
            ky = keypoints[3 * k + 1]
            visibility = keypoints[3 * k + 2]

            # Exclude unlabeled keypoints.
            # Visibility 1 (occluded) and 2 (visible) are retained.
            if visibility == 0:
                continue

            x_norm = kx / img_w
            y_norm = ky / img_h

            kpt_counts[ann["image_id"]] += 1

            kpt_rows.append({
                "image_id": ann["image_id"],
                "kp_index": k,
                "x": kx,
                "y": ky,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "visible_flag": visibility,
                "is_occluded": 1 if visibility == 1 else 0,
                "img_w": img_w,
                "img_h": img_h,
            })

    bbox_columns = [
        "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "area", "aspect_ratio", "img_w", "img_h",
    ]

    kpt_columns = [
        "image_id", "kp_index", "x", "y",
        "x_norm", "y_norm", "visible_flag",
        "is_occluded", "img_w", "img_h",
    ]

    df_bbox = pd.DataFrame(bbox_rows, columns=bbox_columns)
    df_kpt = pd.DataFrame(kpt_rows, columns=kpt_columns)

    bbox_count_series = pd.Series(
        bbox_counts, name="bbox_count", dtype="int64"
    )

    kpt_count_series = pd.Series(
        kpt_counts, name="keypoint_count", dtype="int64"
    )

    return df_bbox, df_kpt, bbox_count_series, kpt_count_series


# ============================================================
# 5. Plotting and CSV export
# ============================================================

def plot_and_save(
    df_bbox: pd.DataFrame,
    df_kpt: pd.DataFrame,
    bbox_counts: pd.Series,
    kpt_counts: pd.Series,
    output_dir: str,
    max_samples_kde: int,
    no_bbox_plot: bool,
    no_kpt_plot: bool,
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 11
    plt.rcParams["font.family"] = "Times New Roman"

    sns.set_style("white")

    rng = np.random.default_rng(0)
    max_n = max(max_samples_kde, 1)

    # --------------------------------------------------------
    # 5.1 Bounding box dimensions KDE
    # --------------------------------------------------------
    if not no_bbox_plot:
        if df_bbox.empty:
            print("No valid bounding boxes; bbox plot skipped.")
        else:
            if len(df_bbox) > max_n:
                indices = rng.choice(
                    len(df_bbox), size=max_n, replace=False
                )
                df_bbox_plot = df_bbox.iloc[indices]
            else:
                df_bbox_plot = df_bbox

            fig_bbox, ax_bbox = plt.subplots(figsize=(5, 4))

            sns.kdeplot(
                data=df_bbox_plot,
                x="bbox_w",
                y="bbox_h",
                fill=True,
                levels=10,
                cmap="viridis",
                thresh=0,
                ax=ax_bbox,
                cbar=True,
                cbar_kws={
                    "orientation": "vertical",
                    "fraction": 0.046,
                    "pad": 0.04,
                    "label": "Density",
                    "format": "%.1e",
                },
            )

            ax_bbox.set_xlabel("Bounding box width (pixels)")
            ax_bbox.set_ylabel("Bounding box height (pixels)")

            ax_bbox.spines["top"].set_visible(False)
            ax_bbox.spines["right"].set_visible(False)

            fig_bbox.suptitle(
                "Bounding Box Dimensions", fontsize=14, y=1.02
            )

            fig_bbox.tight_layout()

            fig_bbox.savefig(
                os.path.join(output_dir, "bbox_dimensions.pdf"),
                bbox_inches="tight",
            )

            plt.close(fig_bbox)

    # --------------------------------------------------------
    # 5.2 Keypoint spatial distribution KDE
    # --------------------------------------------------------
    if not no_kpt_plot:
        if df_kpt.empty:
            print("No valid labeled keypoints; keypoint plot skipped.")
        else:
            if len(df_kpt) > max_n:
                indices = rng.choice(
                    len(df_kpt), size=max_n, replace=False
                )
                df_kpt_plot = df_kpt.iloc[indices]
            else:
                df_kpt_plot = df_kpt

            fig_kpt, ax_kpt = plt.subplots(figsize=(5, 4))

            sns.kdeplot(
                data=df_kpt_plot,
                x="x_norm",
                y="y_norm",
                fill=True,
                levels=10,
                cmap="viridis",
                thresh=0,
                ax=ax_kpt,
                cbar=True,
                cbar_kws={
                    "orientation": "vertical",
                    "fraction": 0.046,
                    "pad": 0.04,
                    "label": "Density",
                    "format": "%.1e",
                },
            )

            ax_kpt.set_xlim(0, 1)
            ax_kpt.set_ylim(0, 1)
            ax_kpt.invert_yaxis()

            ax_kpt.set_xlabel("Normalized X")
            ax_kpt.set_ylabel("Normalized Y")

            ax_kpt.spines["top"].set_visible(False)
            ax_kpt.spines["right"].set_visible(False)

            fig_kpt.suptitle(
                "Keypoint Spatial Distribution (normalized)",
                fontsize=14,
                y=1.02,
            )

            fig_kpt.tight_layout()

            fig_kpt.savefig(
                os.path.join(
                    output_dir, "keypoint_spatial_distribution.pdf"
                ),
                bbox_inches="tight",
            )

            plt.close(fig_kpt)

    # --------------------------------------------------------
    # 5.3 Per-image annotation count histograms
    # --------------------------------------------------------
    mean_bboxes = (
        float(bbox_counts.mean()) if len(bbox_counts) else 0.0
    )

    mean_kpts = (
        float(kpt_counts.mean()) if len(kpt_counts) else 0.0
    )

    print(f"Mean number of bboxes per image: {mean_bboxes:.3f}")
    print(f"Mean number of keypoints per image: {mean_kpts:.3f}")

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4), sharey=True
    )

    if len(bbox_counts):
        axes[0].hist(
            bbox_counts,
            bins=30,
            color="#4C72B0",
            edgecolor="black",
            alpha=0.8,
        )

    axes[0].axvline(
        mean_bboxes,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_bboxes:.2f}",
    )

    axes[0].set_xlabel("Number of bboxes per image")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("Bounding box density")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].legend(frameon=False)

    if len(kpt_counts):
        axes[1].hist(
            kpt_counts,
            bins=30,
            color="#55A868",
            edgecolor="black",
            alpha=0.8,
        )

    axes[1].axvline(
        mean_kpts,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_kpts:.2f}",
    )

    axes[1].set_xlabel("Number of labeled keypoints per image")
    axes[1].set_title("Keypoint density")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        os.path.join(output_dir, "bbox_keypoint_count_hist.pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 5.4 Save CSV statistics
    # --------------------------------------------------------
    bbox_csv = os.path.join(output_dir, "coco_bbox_stats.csv")
    kpt_csv = os.path.join(output_dir, "coco_keypoint_stats.csv")

    df_bbox.to_csv(bbox_csv, index=False)
    df_kpt.to_csv(kpt_csv, index=False)

    # Export per-image counts as an additional useful CSV.
    count_csv = os.path.join(output_dir, "per_image_counts.csv")

    per_image_df = pd.DataFrame({
        "image_id": bbox_counts.index,
        "bbox_count": bbox_counts.values,
        "keypoint_count": kpt_counts.reindex(
            bbox_counts.index, fill_value=0
        ).values,
    })

    per_image_df.to_csv(count_csv, index=False)

    print(f"BBox statistics saved to: {bbox_csv}")
    print(f"Keypoint statistics saved to: {kpt_csv}")
    print(f"Per-image counts saved to: {count_csv}")
    print(f"All outputs saved to: {os.path.abspath(output_dir)}")


# ============================================================
# 6. Main
# ============================================================


def main() -> None:
    args = parse_args()

    if args.format == "coco":
        if not args.coco_json:
            raise ValueError(
                "--coco_json is required when --format coco."
            )

        coco = load_coco_dataset(args.coco_json)

        output_dir = (
            args.output_dir
            or os.path.dirname(os.path.abspath(args.coco_json))
        )

    elif args.format == "yolo":
        if not args.dataset_dir:
            raise ValueError(
                "--dataset_dir is required when --format yolo."
            )

        dataset_dir = Path(args.dataset_dir)
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}"
            )

        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Labels directory not found: {labels_dir}"
            )

        coco = load_yolo_pose_dataset(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            kpt_dims=args.kpt_dims,
        )

        output_dir = (
            args.output_dir
            or os.path.abspath("./yolo_statistics")
        )

    os.makedirs(output_dir, exist_ok=True)

    df_bbox, df_kpt, bbox_counts, kpt_counts = build_statistics(coco)

    print(f"Number of images: {len(coco['images'])}")
    print(f"Number of valid bounding boxes: {len(df_bbox)}")
    print(f"Number of labeled keypoints: {len(df_kpt)}")

    if df_bbox.empty:
        raise ValueError(
            "No valid bounding boxes found. "
            "Check the annotation files and coordinate format."
        )

    plot_and_save(
        df_bbox=df_bbox,
        df_kpt=df_kpt,
        bbox_counts=bbox_counts,
        kpt_counts=kpt_counts,
        output_dir=output_dir,
        max_samples_kde=args.max_samples_kde,
        no_bbox_plot=args.no_bbox_plot,
        no_kpt_plot=args.no_kpt_plot,
    )


if __name__ == "__main__":
    main()
