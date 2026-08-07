import os
import json
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for COCO keypoint dataset statistics."""

    parser = argparse.ArgumentParser(
        description="Compute statistics and visualizations for a COCO keypoint dataset.",
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        required=True,
        help="Path to COCO annotations JSON file (person keypoints).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save figures and CSVs. "
            "Default: same directory as the COCO JSON file."
        ),
    )
    parser.add_argument(
        "--max_samples_kde",
        type=int,
        default=50000,
        help=(
            "Maximum number of boxes/keypoints used for KDE plots. "
            "Useful to speed up visualization on very large datasets."
        ),
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


def main() -> None:
    args = parse_args()

    coco_json_path = args.coco_json
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(coco_json_path))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load COCO annotations and build DataFrames
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    bbox_rows: list[dict] = []
    kpt_rows: list[dict] = []

    print("Parsing annotations and building bbox/keypoint statistics...")
    for ann in coco["annotations"]:
        img_info = images_by_id.get(ann["image_id"])
        if img_info is None:
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        # 1) Bounding box info
        x, y, w, h = ann["bbox"]  # COCO: [x, y, w, h]
        if w <= 0 or h <= 0:
            continue

        bbox_rows.append(
            {
                "image_id": ann["image_id"],
                "category_id": ann.get("category_id", -1),
                "bbox_w": w,
                "bbox_h": h,
                "area": w * h,
                "aspect_ratio": w / h,
                "img_w": img_w,
                "img_h": img_h,
            },
        )

        # 2) Keypoint info ([x1, y1, v1, x2, y2, v2, ...])
        kpts = ann.get("keypoints", [])
        if not kpts:
            continue

        num_kpts = len(kpts) // 3
        for k in range(num_kpts):
            kx = kpts[3 * k]
            ky = kpts[3 * k + 1]
            v = kpts[3 * k + 2]  # 0: not labeled, 1: labeled but not visible, 2: visible

            # Skip non-existing keypoints (v == 0); keep visible/occluded ones only
            if v == 0:
                continue

            # Normalized coordinates (0~1) for spatial distribution plot
            x_norm = kx / img_w
            y_norm = ky / img_h

            kpt_rows.append(
                {
                    "image_id": ann["image_id"],
                    "kp_index": k,
                    "x": kx,
                    "y": ky,
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "visible_flag": v,  # 1=occluded, 2=visible (COCO convention)
                    "is_occluded": 1 if v == 1 else 0,
                    "img_w": img_w,
                    "img_h": img_h,
                },
            )

    df_bbox = pd.DataFrame(bbox_rows)
    df_kpt = pd.DataFrame(kpt_rows)

    if df_bbox.empty:
        raise ValueError(
            "No valid bbox annotations for statistics (annotations may be empty or bbox has zero area).",
        )
    if df_kpt.empty:
        raise ValueError(
            "No valid keypoint annotations for statistics (keypoints may be empty or all v==0).",
        )

    print(f"Number of bboxes: {len(df_bbox)}")
    print(f"Number of keypoints: {len(df_kpt)}")

    # Optionally subsample for KDE plots to speed up visualization on large datasets
    max_n = max(args.max_samples_kde, 1)
    df_bbox_plot = df_bbox.sample(n=max_n, random_state=0) if len(df_bbox) > max_n else df_bbox
    df_kpt_plot = df_kpt.sample(n=max_n, random_state=0) if len(df_kpt) > max_n else df_kpt

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'Times New Roman'

    # Clean, publication-style appearance
    sns.set_style("white")

    # 2. Bounding box dimensions (width x height)
    if not args.no_bbox_plot:
        fig_bbox, ax_bbox = plt.subplots(figsize=(5, 4))
        kde_bbox = sns.kdeplot(
            data=df_bbox_plot,
            x='bbox_w',
            y='bbox_h',
            fill=True,
            levels=10,
            cmap='viridis',
            thresh=0,
            ax=ax_bbox,
            # Let seaborn create a colorbar that matches the colormap used
            cbar=True,
            cbar_kws={
                'orientation': 'vertical',
                'fraction': 0.046,
                'pad': 0.04,
                'label': 'Density',
                'format': '%.1e',
            },
        )

        ax_bbox.set_xlabel("Bounding box width (pixels)")
        ax_bbox.set_ylabel("Bounding box height (pixels)")

        for spine in ["top", "right"]:
            ax_bbox.spines[spine].set_visible(False)

        fig_bbox.suptitle('Bounding Box Dimensions', fontsize=14, y=1.02)
        fig_bbox.tight_layout()
        base_bbox_path = os.path.join(output_dir, 'bbox_dimensions')
        fig_bbox.savefig(base_bbox_path + '.pdf', bbox_inches='tight')

    # 3. Keypoint spatial distribution (normalized coordinates)
    if not args.no_kpt_plot:
        fig_kpt, ax_kpt = plt.subplots(figsize=(5, 4))
        kde_kpt = sns.kdeplot(
            data=df_kpt_plot,
            x='x_norm',
            y='y_norm',
            fill=True,
            levels=10,
            cmap='viridis',
            thresh=0,
            ax=ax_kpt,
            # Let seaborn create a colorbar that matches the colormap used
            cbar=True,
            cbar_kws={
                'orientation': 'vertical',
                'fraction': 0.046,
                'pad': 0.04,
                'label': 'Density',
                'format': '%.1e',
            },
        )

        ax_kpt.set_xlim(0, 1)
        ax_kpt.set_ylim(0, 1)
        ax_kpt.invert_yaxis()

        ax_kpt.set_xlabel("Normalized X")
        ax_kpt.set_ylabel("Normalized Y")

        for spine in ["top", "right"]:
            ax_kpt.spines[spine].set_visible(False)

        fig_kpt.suptitle('Keypoint Spatial Distribution (normalized)', fontsize=14, y=1.02)
        fig_kpt.tight_layout()
        base_kpt_path = os.path.join(output_dir, 'keypoint_spatial_distribution')
        fig_kpt.savefig(base_kpt_path + '.pdf', bbox_inches='tight')

    # 4. Per-image bbox/keypoint density (histograms)
    #    - Combined figure: left = number of bboxes per image
    #                       right = number of keypoints per image
    bbox_counts = df_bbox.groupby('image_id')['bbox_w'].size()
    kpt_counts = df_kpt.groupby('image_id')['kp_index'].size()

    # Per-image mean counts
    mean_bboxes_per_image = float(bbox_counts.mean())
    mean_kpts_per_image = float(kpt_counts.mean())

    print(f"Mean number of bboxes per image: {mean_bboxes_per_image:.3f}")
    print(f"Mean number of keypoints per image: {mean_kpts_per_image:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    axes[0].hist(bbox_counts, bins=30, color="#4C72B0", edgecolor='black', alpha=0.8)
    axes[0].axvline(mean_bboxes_per_image, color='red', linestyle='--', linewidth=1.5,
                    label=f"Mean = {mean_bboxes_per_image:.2f}")
    axes[0].set_xlabel('Number of bboxes per image')
    axes[0].set_ylabel('Number of images')
    axes[0].set_title('Bounding box density')
    axes[0].grid(axis='y', linestyle=':', alpha=0.4)
    axes[0].legend(frameon=False)

    axes[1].hist(kpt_counts, bins=30, color="#55A868", edgecolor='black', alpha=0.8)
    axes[1].axvline(mean_kpts_per_image, color='red', linestyle='--', linewidth=1.5,
                    label=f"Mean = {mean_kpts_per_image:.2f}")
    axes[1].set_xlabel('Number of keypoints per image')
    axes[1].set_title('Keypoint density')
    axes[1].grid(axis='y', linestyle=':', alpha=0.4)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    base_combined_hist = os.path.join(output_dir, 'bbox_keypoint_count_hist')
    fig.savefig(base_combined_hist + '.pdf', bbox_inches='tight')
    plt.close(fig)

    # 5. Save statistics to CSV
    df_bbox.to_csv(os.path.join(output_dir, 'coco_bbox_stats.csv'), index=False)
    df_kpt.to_csv(os.path.join(output_dir, 'coco_keypoint_stats.csv'), index=False)

    print(f"BBox statistics saved to {os.path.join(output_dir, 'coco_bbox_stats.csv')}")
    print(f"Keypoint statistics saved to {os.path.join(output_dir, 'coco_keypoint_stats.csv')}")


if __name__ == '__main__':
    main()