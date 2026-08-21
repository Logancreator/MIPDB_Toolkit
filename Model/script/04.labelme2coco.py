import os
import json
import base64
import argparse
import shutil
import random
import datetime
import logging
from io import BytesIO
from typing import List, Tuple, Dict

from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_skeleton(num_keypoints: int) -> List[List[int]]:
    """Generate skeleton connections for keypoints (0-based indexing)."""
    if num_keypoints <= 1:
        return []
    return [[i, i + 1] for i in range(num_keypoints - 1)]


def get_default_categories(num_keypoints: int = 17) -> Tuple[List[dict], Dict[str, dict]]:
    """Return default categories list and a lookup dict for category info."""
    keypoint_names = [f"kpt_{i + 1}" for i in range(num_keypoints)]
    skeleton = generate_skeleton(num_keypoints)

    categories = [
        {
            "id": 1,
            "name": "stem",
            "supercategory": "corn",
            "keypoints": keypoint_names,
            "skeleton": skeleton,
        },
        {
            "id": 2,
            "name": "leaf",
            "supercategory": "corn",
            "keypoints": keypoint_names,
            "skeleton": skeleton,
        },
    ]

    lookup: Dict[str, dict] = {
        "stem": {"id": 1, "num_keypoints": num_keypoints},
        "leaf": {"id": 2, "num_keypoints": num_keypoints},
    }
    return categories, lookup


def load_json(file_path: str) -> dict:
    """Load a JSON file with basic logging."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading JSON from {file_path}: {e}")
        raise


def save_json(data: dict, file_path: str) -> None:
    """Save data to a JSON file with basic logging."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved JSON to: {file_path}")
    except IOError as e:
        logger.error(f"Could not write JSON to {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving JSON to {file_path}: {e}")
        raise


def get_bbox_from_points(points: List[List[float]], expansion: int = 0) -> List[float]:
    """Calculate TLWH bbox from points with optional expansion."""
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, y_min = min(x_coords) - expansion, min(y_coords) - expansion
    x_max, y_max = max(x_coords) + expansion, max(y_coords) + expansion
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    return [x_min, y_min, width, height]


def parse_label(label: str) -> Tuple[str, str]:
    """Parse LabelMe label to determine category and corn_id."""
    if "-y" in label:
        corn_id = label.split("-y")[0]
        return "leaf", corn_id
    return "stem", label


def labelme_to_coco_single(
    labelme_data: dict,
    bbox_expansion: int,
    image_id: int,
    annotation_start_id: int,
    category_lookup: Dict[str, dict],
) -> Tuple[dict, List[dict], int]:
    """Convert one LabelMe JSON data into COCO image dict and list of annotations."""

    image_path = labelme_data.get("imagePath", f"unknown_{image_id}.jpg")
    image_info = {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "height": labelme_data.get("imageHeight", 0),
        "width": labelme_data.get("imageWidth", 0),
        "license": 1,
        "date_captured": "",
    }

    annotations: List[dict] = []
    ann_id = annotation_start_id

    for shape in labelme_data.get("shapes", []):
        if shape.get("shape_type") != "linestrip" or not shape.get("label"):
            logger.debug(
                f"Skipping shape with type '{shape.get('shape_type')}' or missing label",
            )
            continue

        category_name, corn_id = parse_label(shape["label"])
        if category_name not in ["stem", "leaf"]:
            logger.warning(
                f"Skipping unknown category '{category_name}' for label '{shape.get('label')}')",
            )
            continue

        points = shape.get("points", [])
        visibility = shape.get("visibility", [1] * len(points))

        if not points:
            logger.warning(
                f"Skipping annotation for label '{shape.get('label')}' due to empty points",
            )
            continue

        new_category_name = category_name
        if new_category_name not in category_lookup:
            logger.warning(
                f"Skipping unknown category '{new_category_name}' for label '{shape.get('label')}'",
            )
            continue
        category_info = category_lookup[new_category_name]

        bbox = get_bbox_from_points(points, bbox_expansion)
        area = bbox[2] * bbox[3]

        keypoints_coco: List[float] = []
        num_labeled_keypoints = 0
        for i in range(category_info["num_keypoints"]):
            if i < len(points):
                x, y = points[i]
                if i >= len(visibility):
                    v = 2
                else:
                    lab_v = visibility[i]
                    if lab_v == 1:
                        v = 2
                    elif lab_v == 0:
                        v = 1
                    else:
                        v = 2

                keypoints_coco.extend([
                    round(float(x), 2),
                    round(float(y), 2),
                    int(v),
                ])
                if v > 0:
                    num_labeled_keypoints += 1
            else:
                keypoints_coco.extend([0.0, 0.0, 0])

        ann = {
            "id": ann_id,
            "image_id": image_info["id"],
            "category_id": category_info["id"],
            "corn_id": corn_id,
            "segmentation": [],
            "area": round(area, 2),
            "bbox": [round(float(coord), 2) for coord in bbox],
            "iscrowd": 0,
            "keypoints": keypoints_coco,
            "num_keypoints": num_labeled_keypoints,
        }

        annotations.append(ann)
        ann_id += 1

    return image_info, annotations, ann_id


def convert_folder_to_coco(
    input_dir: str,
    output_path: str,
    bbox_expansion: int = 0,
    num_keypoints: int = 17,
) -> bool:
    """Convert a folder of LabelMe JSON files into a single merged COCO JSON."""

    try:
        logger.info(f"Converting folder to merged COCO: {input_dir} -> {output_path}")

        coco_output = {
            "info": {
                "description": "Corn plant dataset for computer vision tasks",
                "url": "http://phenomics.agis.org.cn/#/category",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "Jianye Chang",
                "date_created": datetime.datetime.now().strftime("%Y/%m/%d"),
            },
            "licenses": [{"url": "", "id": 1, "name": "Default License"}],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        categories, lookup = get_default_categories(num_keypoints)
        coco_output["categories"] = categories

        image_id = 1
        ann_id = 1

        files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        files.sort()
        for fname in files:
            path = os.path.join(input_dir, fname)
            try:
                data = load_json(path)
            except Exception as e:  # noqa: PERF203
                logger.error(f"Failed to load {path}: {e}")
                continue

            img_info, anns, next_ann_id = labelme_to_coco_single(
                data,
                bbox_expansion,
                image_id,
                ann_id,
                lookup,
            )
            if anns:
                coco_output["images"].append(img_info)
                coco_output["annotations"].extend(anns)
                image_id += 1
                ann_id = next_ann_id
            else:
                logger.info(f"No valid annotations found in {fname}; skipping adding image")

        save_json(coco_output, output_path)
        logger.info(f"Saved merged COCO to {output_path}")
        return True
    except Exception as e:  # noqa: PERF203
        logger.error(f"Failed to convert folder to COCO: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for Labelme -> COCO conversion.

    This is a lightweight wrapper around the existing labelme2coco_script
    conversion utilities. It merges一个文件夹里的所有 Labelme JSON 为一个
    COCO annotations.json，并支持 bbox 扩张像素设置。
    """
    parser = argparse.ArgumentParser(
        description="Convert Labelme JSON annotations to a single COCO JSON file.",
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="Directory containing Labelme JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output COCO JSON path (e.g., ./output/annotations.json).",
    )
    parser.add_argument(
        "--bbox_expansion",
        type=int,
        default=25,
        help="Bounding box expansion in pixels around keypoints (default: 25).",
    )
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=17,
        help="Number of keypoints per instance (must match your data, default: 17).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help=(
            "Directory to save all images referenced by COCO annotations. "
            "If omitted, defaults to '<output_parent>/images'."
        ),
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset split (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    json_dir = args.json_dir
    output_path = args.output

    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"json_dir does not exist or is not a directory: {json_dir}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Determine unified images directory (all COCO images go here)
    images_dir = args.images_dir
    if images_dir is None:
        images_dir = os.path.join(out_dir or ".", "images")
    os.makedirs(images_dir, exist_ok=True)

    # Prepare a temporary directory with cleaned JSONs whose imagePath
    # matches the saved image file name. This avoids修改原始 Labelme 文件。
    temp_json_dir = os.path.join(out_dir or ".", "_tmp_labelme_for_coco")
    if os.path.exists(temp_json_dir):
        shutil.rmtree(temp_json_dir)
    os.makedirs(temp_json_dir, exist_ok=True)

    # 1) For each Labelme JSON, decode imageData (if present) to the
    #    unified images_dir, and write a cleaned JSON copy to temp_json_dir.
    for fname in os.listdir(json_dir):
        if not fname.lower().endswith(".json"):
            continue

        src_json_path = os.path.join(json_dir, fname)
        with open(src_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base_name = os.path.splitext(os.path.basename(fname))[0]
        img_name = base_name + ".JPG"
        img_output_path = os.path.join(images_dir, img_name)

        # Decode embedded base64 image if available; otherwise just rely on
        # existing files on disk with the same name (if user provides them).
        image_data = data.get("imageData")
        if image_data:
            try:
                img_bytes = base64.b64decode(image_data)
                img = Image.open(BytesIO(img_bytes))
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(img_output_path)
            except Exception as e:
                print(f"Warning: failed to decode/save image for {src_json_path}: {e}")

        # Update imagePath to the unified image file name so that COCO
        # 'file_name' 字段与实际图片文件名一致。
        data["imagePath"] = img_name

        # Ensure imageWidth/Height exist if we just saved the image
        if (data.get("imageWidth") is None or data.get("imageHeight") is None) and os.path.exists(img_output_path):
            try:
                with Image.open(img_output_path) as im:
                    w, h = im.size
                data["imageWidth"] = w
                data["imageHeight"] = h
            except Exception as e:
                print(f"Warning: cannot read size for {img_output_path}: {e}")

        dst_json_path = os.path.join(temp_json_dir, fname)
        with open(dst_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 2) Merge cleaned JSONs into a single COCO annotations file.
    success = convert_folder_to_coco(
        input_dir=temp_json_dir,
        output_path=output_path,
        bbox_expansion=args.bbox_expansion,
        num_keypoints=args.num_keypoints,
    )

    if not success:
        raise SystemExit("Failed to convert Labelme folder to COCO format.")
    # Optional: clean up temporary JSON directory after successful conversion
    shutil.rmtree(temp_json_dir, ignore_errors=True)

    # 3) Split COCO dataset into train/val/test subsets.
    #    - Randomly split images list by image_id
    #    - Filter annotations per split
    #    - Copy images into images/train|val|test
    #    - Save annotations_train/val/test.json

    # Load merged COCO
    with open(output_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    if not images:
        print("Warning: merged COCO has no images; skipping split.")
        return

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if total_ratio <= 0:
        raise ValueError("Sum of train/val/test ratios must be > 0")

    # Normalize ratios
    train_r = args.train_ratio / total_ratio
    val_r = args.val_ratio / total_ratio
    test_r = args.test_ratio / total_ratio

    random.seed(args.seed)
    shuffled_images = images[:]
    random.shuffle(shuffled_images)

    n_total = len(shuffled_images)
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)
    n_test = n_total - n_train - n_val

    train_images = shuffled_images[:n_train]
    val_images = shuffled_images[n_train:n_train + n_val]
    test_images = shuffled_images[n_train + n_val:]

    split_map = {}
    for img in train_images:
        split_map[img["id"]] = "train"
    for img in val_images:
        split_map[img["id"]] = "val"
    for img in test_images:
        split_map[img["id"]] = "test"

    # Group annotations by split via image_id
    split_images = {"train": train_images, "val": val_images, "test": test_images}
    split_annotations = {"train": [], "val": [], "test": []}
    for ann in annotations:
        img_id = ann.get("image_id")
        split = split_map.get(img_id)
        if split is not None:
            split_annotations[split].append(ann)

    # Prepare image subdirectories in COCO-style naming (train2017/val2017/test2017)
    split_image_dirs = {
        "train": os.path.join(images_dir, "train2017"),
        "val": os.path.join(images_dir, "val2017"),
        "test": os.path.join(images_dir, "test2017"),
    }
    for d in split_image_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Annotations directory with COCO-style filenames
    ann_dir = os.path.join(out_dir or ".", "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    # Move/copy images into subfolders and write split annotations
    for split_name in ["train", "val", "test"]:
        imgs = split_images[split_name]
        anns = split_annotations[split_name]

        # Copy images
        for img in imgs:
            file_name = img.get("file_name")
            if not file_name:
                continue
            src_img_path = os.path.join(images_dir, file_name)
            dst_img_path = os.path.join(split_image_dirs[split_name], file_name)

            if os.path.exists(src_img_path):
                if not os.path.exists(dst_img_path):
                    try:
                        shutil.copy2(src_img_path, dst_img_path)
                    except Exception as e:
                        print(f"Warning: failed to copy {src_img_path} to {dst_img_path}: {e}")
            else:
                print(f"Warning: image file not found for {file_name} when copying to {split_name} set")

        # Build COCO dict for this split
        split_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": imgs,
            "annotations": anns,
            "categories": coco.get("categories", []),
        }

        if split_name == "train":
            ann_filename = "person_keypoints_train2017.json"
        elif split_name == "val":
            ann_filename = "person_keypoints_val2017.json"
        else:
            ann_filename = "person_keypoints_test2017.json"

        split_ann_path = os.path.join(ann_dir, ann_filename)
        with open(split_ann_path, "w", encoding="utf-8") as f:
            json.dump(split_coco, f, ensure_ascii=False, indent=2)

    print(
        f"COCO keypoint dataset split complete: "
        f"train={len(train_images)}, val={len(val_images)}, test={len(test_images)}"
    )


if __name__ == "__main__":
    main()
