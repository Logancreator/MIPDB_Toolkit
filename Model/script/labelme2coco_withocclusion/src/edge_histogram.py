import os
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import load_json, logger


EDGE_LABELS = {
    "OO": "Occluded - Occluded",
    "OV": "Occluded - Visible",
    "VO": "Visible - Occluded",
    "VV": "Visible - Visible",
}


def _collect_edge_counts(
    coco: dict,
    ignore_unlabeled: bool = True,
) -> Dict[str, Dict]:
    """Collect edge counts by occlusion pair type.

    Returns a dict with:
    - totals: {"OO": int, "OV": int, "VO": int, "VV": int}
    - per_image: {image_id: {"OO": int, "OV": int, "VO": int, "VV": int}}
    """
    cat_skeleton: Dict[int, List[Tuple[int, int]]] = {}
    for cat in coco.get("categories", []):
        cat_id = cat.get("id")
        skeleton = cat.get("skeleton", []) or []
        cat_skeleton[cat_id] = [tuple(conn) for conn in skeleton if len(conn) == 2]

    totals = {"OO": 0, "OV": 0, "VO": 0, "VV": 0}
    per_image: Dict[int, Dict[str, int]] = {}

    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in per_image:
            per_image[img_id] = {"OO": 0, "OV": 0, "VO": 0, "VV": 0}

        skeleton = cat_skeleton.get(ann.get("category_id"), [])
        kpts = ann.get("keypoints", [])
        if not kpts or not skeleton:
            continue

        try:
            kp = np.array(kpts, dtype=float).reshape(-1, 3)
        except Exception:
            continue

        for i, j in skeleton:
            if not (0 <= i < len(kp) and 0 <= j < len(kp)):
                continue
            _, _, v1 = kp[i]
            _, _, v2 = kp[j]

            if ignore_unlabeled and (v1 == 0 or v2 == 0):
                continue
            if v1 not in (1, 2) or v2 not in (1, 2):
                continue

            if v1 == 1 and v2 == 1:
                key = "OO"
            elif v1 == 1 and v2 == 2:
                key = "OV"
            elif v1 == 2 and v2 == 1:
                key = "VO"
            else:  # v1==2 and v2==2
                key = "VV"

            totals[key] += 1
            per_image[img_id][key] += 1

    return {"totals": totals, "per_image": per_image}


def _collect_edge_lengths(
    coco: dict,
    ignore_unlabeled: bool = True,
) -> Dict[str, List[float]]:
    """Collect edge lengths by visibility-pair type from a COCO keypoint dataset.

    Returns a dict mapping {"OO"|"OV"|"VO"|"VV": [lengths...]}
    - v=0: not labeled, v=1: occluded, v=2: visible
    - If ignore_unlabeled is True, skip edges with any endpoint v==0.
    """
    # Prepare category skeletons (0-based indices expected)
    cat_skeleton: Dict[int, List[Tuple[int, int]]] = {}
    for cat in coco.get("categories", []):
        cat_id = cat.get("id")
        skeleton = cat.get("skeleton", []) or []
        # Ensure each is a tuple
        cat_skeleton[cat_id] = [tuple(conn) for conn in skeleton if len(conn) == 2]

    edge_buckets = {"OO": [], "OV": [], "VO": [], "VV": []}

    # Image info is available if normalization is needed (not used here)

    for ann in coco.get("annotations", []):
        cat_id = ann.get("category_id")
        skeleton = cat_skeleton.get(cat_id, [])
        kpts = ann.get("keypoints", [])
        if not kpts or not skeleton:
            continue
        try:
            kp = np.array(kpts, dtype=float).reshape(-1, 3)
        except Exception:
            continue

        for i, j in skeleton:
            if not (0 <= i < len(kp) and 0 <= j < len(kp)):
                continue
            x1, y1, v1 = kp[i]
            x2, y2, v2 = kp[j]

            # skip edges with unlabeled endpoints if requested
            if ignore_unlabeled and (v1 == 0 or v2 == 0):
                continue

            # Only consider occluded/visible states
            if v1 not in (1, 2) or v2 not in (1, 2):
                continue

            length = math.hypot(float(x1) - float(x2), float(y1) - float(y2))

            if v1 == 1 and v2 == 1:
                edge_buckets["OO"].append(length)
            elif v1 == 1 and v2 == 2:
                edge_buckets["OV"].append(length)
            elif v1 == 2 and v2 == 1:
                edge_buckets["VO"].append(length)
            elif v1 == 2 and v2 == 2:
                edge_buckets["VV"].append(length)

    return edge_buckets


def plot_edge_count_histograms(
    coco_json_path: str,
    output_dir: str,
    bins: int = 20,
    ignore_unlabeled: bool = True,
    dpi: int = 180,
) -> Dict[str, Dict]:
    """Plot edge count distributions by occlusion pair.

    - Produces a per-image 2x2 histogram grid: `edge_count_hist_2x2.png`.
    - Produces a totals bar chart: `edge_count_totals.png`.
    - Returns the counts dict from `_collect_edge_counts`.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading COCO annotations: {coco_json_path}")
    coco = load_json(coco_json_path)

    counts = _collect_edge_counts(coco, ignore_unlabeled=ignore_unlabeled)
    totals = counts["totals"]
    per_image = counts["per_image"]

    # 1) Per-image histogram grid
    per_image_lists = {k: [] for k in ["OO", "OV", "VO", "VV"]}
    for img_id, d in per_image.items():
        for k in per_image_lists.keys():
            per_image_lists[k].append(int(d.get(k, 0)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=dpi, constrained_layout=True)
    pairs = [("OO", axes[0, 0]), ("OV", axes[0, 1]), ("VO", axes[1, 0]), ("VV", axes[1, 1])]
    colors = {
        "OO": "#d62728",
        "OV": "#ff7f0e",
        "VO": "#1f77b4",
        "VV": "#2ca02c",
    }
    for key, ax in pairs:
        data = np.array(per_image_lists[key], dtype=float)
        if data.size > 0:
            # bins from 0 to max count inclusive
            max_c = int(np.max(data))
            bin_edges = np.arange(-0.5, max(max_c, 1) + 1.5)  # center on integers
            ax.hist(data, bins=bin_edges, color=colors[key], alpha=0.8, edgecolor="white")
            stats = f"images={len(data)}  mean={data.mean():.2f}  max={int(data.max())}"
        else:
            ax.hist([], bins=10, color=colors[key], alpha=0.8, edgecolor="white")
            stats = "images=0"
        ax.set_title(EDGE_LABELS[key])
        ax.set_xlabel("Edges per image")
        ax.set_ylabel("Image count")
        ax.grid(True, ls=":", alpha=0.4)
        ax.text(0.98, 0.95, stats, transform=ax.transAxes, ha="right", va="top", fontsize=9)

    fig.suptitle("Edge Count Histograms per Image (by occlusion pair)", fontsize=14)
    out_png_hist = os.path.join(output_dir, "edge_count_hist_2x2.png")
    fig.savefig(out_png_hist)
    plt.close(fig)
    logger.info(f"Saved per-image count histograms to: {out_png_hist}")

    # 2) Totals bar chart
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5), dpi=dpi, constrained_layout=True)
    keys = ["OO", "OV", "VO", "VV"]
    values = [int(totals[k]) for k in keys]
    bar_colors = [colors[k] for k in keys]
    bars = ax2.bar(keys, values, color=bar_colors)
    for rect, v in zip(bars, values):
        ax2.text(rect.get_x() + rect.get_width() / 2, v, str(v), ha="center", va="bottom")
    ax2.set_title("Total Edge Counts (by occlusion pair)")
    ax2.set_xlabel("Occlusion pair")
    ax2.set_ylabel("Count")
    ax2.grid(True, axis="y", ls=":", alpha=0.4)
    out_png_bar = os.path.join(output_dir, "edge_count_totals.png")
    fig2.savefig(out_png_bar)
    plt.close(fig2)
    logger.info(f"Saved totals bar chart to: {out_png_bar}")

    # CSV summary (totals + per-image rows)
    try:
        import csv
        out_csv = os.path.join(output_dir, "edge_count_stats.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Level", "ImageID", "OO", "OV", "VO", "VV", "Sum"])
            writer.writerow(["TOTAL", "-", totals["OO"], totals["OV"], totals["VO"], totals["VV"], sum(values)])
            for img_id, d in sorted(per_image.items(), key=lambda x: x[0] if x[0] is not None else -1):
                row_vals = [int(d.get(k, 0)) for k in keys]
                writer.writerow(["IMAGE", img_id, *row_vals, sum(row_vals)])
        logger.info(f"Saved count stats CSV to: {out_csv}")
    except Exception as e:
        logger.warning(f"Failed writing count CSV: {e}")

    return counts


def plot_edge_histograms(
    coco_json_path: str,
    output_dir: str,
    bins: int = 30,
    ignore_unlabeled: bool = True,
    dpi: int = 180,
) -> Dict[str, List[float]]:
    """Plot 2x2 histograms of edge-length distributions by occlusion pairs.

    - Saves `edge_length_hist_2x2.png` to `output_dir`.
    - Returns the raw buckets for further analysis or testing.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading COCO annotations: {coco_json_path}")
    coco = load_json(coco_json_path)

    buckets = _collect_edge_lengths(coco, ignore_unlabeled=ignore_unlabeled)

    # Flatten to get global range for consistent bins
    all_lengths = np.concatenate([np.array(v, dtype=float) for v in buckets.values() if len(v) > 0], axis=0) if any(len(v) for v in buckets.values()) else np.array([])

    if all_lengths.size == 0:
        logger.warning("No edge lengths found to plot. Skipping figure generation.")
        return buckets

    vmin, vmax = float(np.nanmin(all_lengths)), float(np.nanmax(all_lengths))
    if vmax <= vmin:
        vmax = vmin + 1.0
    bin_edges = np.linspace(vmin, vmax, bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=dpi, constrained_layout=True)
    pairs = [
        ("OO", axes[0, 0]),
        ("OV", axes[0, 1]),
        ("VO", axes[1, 0]),
        ("VV", axes[1, 1]),
    ]

    colors = {
        "OO": "#d62728",  # red-ish
        "OV": "#ff7f0e",  # orange
        "VO": "#1f77b4",  # blue
        "VV": "#2ca02c",  # green
    }

    for key, ax in pairs:
        data = np.array(buckets.get(key, []), dtype=float)
        label_text = EDGE_LABELS[key]
        if data.size > 0:
            ax.hist(data, bins=bin_edges, color=colors[key], alpha=0.75, edgecolor="white")
            stats = f"n={len(data)}  mean={data.mean():.1f}  std={data.std(ddof=1) if len(data) > 1 else 0:.1f}"
        else:
            ax.hist([], bins=bin_edges, color=colors[key], alpha=0.75, edgecolor="white")
            stats = "n=0"
        ax.set_title(f"{label_text}")
        ax.set_xlabel("Length (pixels)")
        ax.set_ylabel("Count")
        ax.grid(True, ls=":", alpha=0.4)
        ax.text(0.98, 0.95, stats, transform=ax.transAxes, ha="right", va="top", fontsize=9)

    fig.suptitle("Keypoint Edge-Length Histograms (by occlusion pair)", fontsize=14)

    out_png = os.path.join(output_dir, "edge_length_hist_2x2.png")
    fig.savefig(out_png)
    plt.close(fig)
    logger.info(f"Saved histogram figure to: {out_png}")

    try:
        import csv
        out_csv = os.path.join(output_dir, "edge_length_stats.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Code", "Count", "Mean", "Median", "P90", "Min", "Max"])
            for key in ["OO", "OV", "VO", "VV"]:
                data = np.array(buckets.get(key, []), dtype=float)
                if data.size > 0:
                    writer.writerow([
                        EDGE_LABELS[key],
                        key,
                        len(data),
                        f"{data.mean():.3f}",
                        f"{np.median(data):.3f}",
                        f"{np.percentile(data, 90):.3f}",
                        f"{data.min():.3f}",
                        f"{data.max():.3f}",
                    ])
                else:
                    writer.writerow([EDGE_LABELS[key], key, 0, "-", "-", "-", "-", "-"])
        logger.info(f"Saved histogram stats CSV to: {out_csv}")
    except Exception as e:
        logger.warning(f"Failed writing CSV summary: {e}")

    return buckets


if __name__ == "__main__":
    # Default: read config.json and use merged annotations + output_dir
    import argparse
    import json

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo_root, "config.json")

    default_coco = None
    default_out = None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            default_out = cfg.get("output_dir")
            if default_out:
                default_coco = os.path.join(default_out, "annotations.json")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Plot occlusion-based edge distributions from COCO annotations.")
    parser.add_argument("--coco", type=str, default=default_coco, help="Path to COCO annotations.json")
    parser.add_argument("--out", type=str, default=default_out, help="Directory to save plots and stats")
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins")
    parser.add_argument("--keep-unlabeled", action="store_true", help="Include edges with unlabeled endpoints (v=0)")
    parser.add_argument("--mode", choices=["count", "length"], default="count", help="Plot edge count distributions or edge length distributions")

    args = parser.parse_args()

    if not args.coco or not os.path.isfile(args.coco):
        raise SystemExit(f"COCO file not found: {args.coco}")
    if not args.out:
        raise SystemExit("Output directory not specified and not found in config.json")

    if args.mode == "count":
        plot_edge_count_histograms(
            coco_json_path=args.coco,
            output_dir=args.out,
            bins=args.bins,
            ignore_unlabeled=not args.keep_unlabeled,
        )
    else:
        plot_edge_histograms(
            coco_json_path=args.coco,
            output_dir=args.out,
            bins=args.bins,
            ignore_unlabeled=not args.keep_unlabeled,
        )
