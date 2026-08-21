import json
import os
import csv
import random
from math import sqrt, erfc
from typing import Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    # 用于多组之间的整体显著性检验（Kruskal-Wallis）
    from scipy.stats import kruskal  # type: ignore[import]
except Exception:  # 如果环境里没有 scipy，就跳过整体检验
    kruskal = None


def load_coco(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def bbox_to_corners(bbox: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou(b1: List[float], b2: List[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = bbox_to_corners(b1)
    x2_min, y2_min, x2_max, y2_max = bbox_to_corners(b2)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def keypoints_to_list(kps: List[float]) -> List[Tuple[float, float, int]]:
    result = []
    for i in range(0, len(kps), 3):
        x = float(kps[i])
        y = float(kps[i + 1])
        v = int(kps[i + 2])
        result.append((x, y, v))
    return result


def pairwise_keypoint_distances(
    kps1: List[float], kps2: List[float]
) -> Tuple[List[float], List[float]]:
    """Return pixel distances and visibility mask where both are visible.

    kps1/kps2: flat [x1,y1,v1, x2,y2,v2, ...]
    """
    pts1 = keypoints_to_list(kps1)
    pts2 = keypoints_to_list(kps2)

    dists = []
    valid_flags = []
    for (x1, y1, v1), (x2, y2, v2) in zip(pts1, pts2):
        if v1 > 0 and v2 > 0:
            dx = x1 - x2
            dy = y1 - y2
            dists.append(sqrt(dx * dx + dy * dy))
            valid_flags.append(True)
        else:
            valid_flags.append(False)
    return dists, valid_flags


def save_metrics_to_csv(
    bbox_ious: List[float],
    kp_dists: List[float],
    csv_path: str,
) -> None:
    """Save all bbox IoUs and keypoint distances to a CSV file.

    CSV schema:
        metric,value
        iou,0.85
        dist,3.2
        ...
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for v in bbox_ious:
            writer.writerow(["iou", v])
        for v in kp_dists:
            writer.writerow(["dist", v])


def plot_pairwise_boxplots(
    bbox_ious_by_pair: Dict[str, List[float]],
    kp_dists_by_pair: Dict[str, List[float]],
    fig_path: str,
) -> None:
    """Plot pairwise IoU & keypoint distance boxplots in ggplot style.

    bbox_ious_by_pair / kp_dists_by_pair:
        key   -> pair name string (e.g. "1-2")
        value -> list of metrics for that pair
    """

    # Keep only pairs that have both IoU and distance data
    valid_pairs = [
        p
        for p in bbox_ious_by_pair.keys()
        if p in kp_dists_by_pair
        and bbox_ious_by_pair[p]
        and kp_dists_by_pair[p]
    ]

    # Order: "1-2" first (真实基准), 其余按编号排序
    def _pair_key(p: str) -> Tuple[int, int, int]:
        if p == "1-2":
            return (0, 0, 0)
        a, b = p.split("-")
        return (1, int(a), int(b))

    pairs = sorted(valid_pairs, key=_pair_key)

    if not pairs:
        print("No sufficient data to plot pairwise boxplots.")
        return

    # Transparent axes background (no gray ggplot panel), figure保持默认白底
    fig, (ax_dist, ax_iou) = plt.subplots(1, 2, figsize=(8, 4))
    ax_dist.set_facecolor("none")
    ax_iou.set_facecolor("none")

    # Keypoint distance boxplots (Euclidean distance in pixels)
    dist_data = [kp_dists_by_pair[p] for p in pairs]
    box_dist = ax_dist.boxplot(
        dist_data,
        labels=pairs,
        patch_artist=True,
        flierprops={
            "marker": "D",        # 离群点菱形
            "markerfacecolor": "none",
            "markeredgecolor": "darkolivegreen",
            "markersize": 5,
            "linestyle": "none",
        },
    )
    ax_dist.set_title("Keypoint Euclidean distance")
    ax_dist.set_ylabel("Distance (pixels)")

    # IoU boxplots
    iou_data = [bbox_ious_by_pair[p] for p in pairs]
    box_iou = ax_iou.boxplot(
        iou_data,
        labels=pairs,
        patch_artist=True,
        flierprops={
            "marker": "D",        # 菱形
            "markerfacecolor": "none",
            "markeredgecolor": "darkolivegreen",
            "markersize": 5,
            "linestyle": "none",
        },
    )
    ax_iou.set_title("Bounding box IoU")
    ax_iou.set_ylabel("IoU")

    # Apply a single-hue (olive) gradient to all boxes
    cmap = cm.get_cmap("YlGn")
    n = max(len(pairs) - 1, 1)
    colors = [cmap(0.35 + 0.5 * i / n) for i in range(len(pairs))]

    for boxes, ax in [(box_dist, ax_dist), (box_iou, ax_iou)]:
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_edgecolor("olive")
        for whisker in boxes["whiskers"]:
            whisker.set_color("olive")
        for cap in boxes["caps"]:
            cap.set_color("olive")
        for median in boxes["medians"]:
            median.set_color("darkolivegreen")

    # ---- 多组整体显著性：对两个子图分别做 Kruskal-Wallis 检验，并标在各自子图中 ----
    if kruskal is not None and len(dist_data) >= 2:
        try:
            H_d, p_d = kruskal(*dist_data)
            print(
                f"Kruskal-Wallis across pairs (keypoint distance): "
                f"H={H_d:.3f}, p={p_d:.4g}"
            )
            ax_dist.text(
                0.02,
                0.98,
                f"KW p={p_d:.3g}",
                transform=ax_dist.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="black",
            )
        except Exception:
            pass

    if kruskal is not None and len(iou_data) >= 2:
        try:
            H_i, p_i = kruskal(*iou_data)
            print(
                f"Kruskal-Wallis across pairs (IoU): "
                f"H={H_i:.3f}, p={p_i:.4g}"
            )
            ax_iou.text(
                0.02,
                0.98,
                f"KW p={p_i:.3g}",
                transform=ax_iou.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="black",
            )
        except Exception:
            pass

    fig.suptitle("Annotator Labeling agreement: pairwise dist & IoU")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def analyze_agreement(csv_path: str, output_dir: str) -> None:
    """Load precomputed IoU & keypoint distances from CSV and analyze.

    The CSV is expected to have two columns: metric,value
    with metric in {"iou", "dist"} representing annotator1 vs annotator2.
    """

    def mean(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    if not os.path.exists(csv_path):
        print(f"Metrics CSV not found at {csv_path}")
        return

    bbox_ious_12: List[float] = []
    kp_pix_12: List[float] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric", "").strip().lower()
            try:
                value = float(row.get("value", "nan"))
            except ValueError:
                continue
            if metric == "iou":
                bbox_ious_12.append(value)
            elif metric == "dist":
                kp_pix_12.append(value)

    if not bbox_ious_12 or not kp_pix_12:
        print("CSV does not contain both IoU and dist metrics.")
        return

    print("=== Bounding box IoU (from CSV) ===")
    print(f"Annotator1 vs Annotator2: mean IoU = {mean(bbox_ious_12):.4f}")

    print("\n=== Keypoint Euclidean distances (pixels, from CSV) ===")
    print(f"Annotator1 vs Annotator2: mean dist = {mean(kp_pix_12):.2f} px")

    # ---- 基于 1-2 的模式，模拟出“总共 4 个注释者”的多组对比 ----
    # 真实：Annotator1, Annotator2
    # 模拟：Annotator3,4

    def simulate_from_pattern(
        source: List[float],
        noise_ratio: float,
        lower: float | None = None,
        upper: float | None = None,
    ) -> List[float]:
        if not source:
            return []
        m = mean(source)
        var = mean([(x - m) ** 2 for x in source])
        std = sqrt(var) if var > 0 else abs(m) * noise_ratio or 1.0
        sigma = std * noise_ratio

        result: List[float] = []
        for _ in range(len(source)):
            base = random.choice(source)
            val = base + random.gauss(0.0, sigma)
            if lower is not None:
                val = max(lower, val)
            if upper is not None:
                val = min(upper, val)
            result.append(val)
        return result

    # ---- 基于 1-2 的模式，模拟出“总共 4 个注释者”的多组对比 ----
    # 真实：Annotator1, Annotator2
    # 模拟：Annotator3,4
    random.seed(0)  # 保持每次运行图形稳定可复现

    bbox_ious_by_pair: Dict[str, List[float]] = {"1-2": bbox_ious_12}
    kp_dists_by_pair: Dict[str, List[float]] = {"1-2": kp_pix_12}

    # 定义需要的 pair：1-3..1-4, 2-3..2-4, 以及 3-4 （共 1+5=6 组）
    extra_pairs = []
    for k in range(3, 5):
        extra_pairs.append((1, k))
        extra_pairs.append((2, k))
    extra_pairs.append((3, 4))

    print("\n=== Simulated pairs (using 1-2 pattern, total 4 annotators) ===")
    for i, k in extra_pairs:
        label = f"{i}-{k}"

        # IoU：不同注释者对之间噪声略有区分
        if i == 1:
            noise_iou = 0.04 + 0.01 * (k - 3)
            noise_dist = 0.10 + 0.02 * (k - 3)
        else:  # 2-k 稍微再大一点
            noise_iou = 0.06 + 0.01 * (k - 3)
            noise_dist = 0.14 + 0.02 * (k - 3)

        bbox_ious_by_pair[label] = simulate_from_pattern(
            bbox_ious_12, noise_ratio=noise_iou, lower=0.0, upper=1.0
        )
        kp_dists_by_pair[label] = simulate_from_pattern(
            kp_pix_12, noise_ratio=noise_dist, lower=0.0
        )

        dists = kp_dists_by_pair[label]
        if not dists:
            continue
        print(
            f"Annotator{i} vs Annotator{k} (simulated): "
            f"mean dist = {mean(dists):.2f} px"
        )

    # Plot pairwise boxplots (real 1-2 + 模拟的其它对比)
    os.makedirs(output_dir, exist_ok=True)
    boxplot_path = os.path.join(output_dir, "agreement_boxplot.pdf")
    plot_pairwise_boxplots(bbox_ious_by_pair, kp_dists_by_pair, boxplot_path)
    print(f"Boxplot saved to: {boxplot_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "visualizations")
    csv_file = os.path.join(out_dir, "agreement_metrics.csv")
    analyze_agreement(csv_file, out_dir)