import json
import os
import glob
import shutil
import argparse
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import random


def _save_image_from_base64(image_data, output_path):
    """Decode base64 image data and save to output_path.

    Returns True on success, False otherwise.
    """
    try:
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"Saved image to {output_path} from embedded base64 data")
        return True
    except Exception as e:
        print(f"Error decoding/saving embedded image to {output_path}: {e}")
        return False


def _parse_label(label):
    """Parse Labelme label to extract corn_id.

    Rules:
    - Labels containing '-y' are leaves, but we only need the corn_id
    - Other labels are stems
    Returns (category_type, corn_id).
    """
    if "-y" in label:
        corn_id = label.split("-y")[0]
        return "leaf", corn_id
    return "stem", label


def _get_bbox_from_points(points, expansion=0):
    """Compute [x_min, y_min, w, h] bbox from points, with optional expansion."""
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min = min(x_coords) - expansion
    y_min = min(y_coords) - expansion
    x_max = max(x_coords) + expansion
    y_max = max(y_coords) + expansion
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    return [x_min, y_min, w, h]


def convert_labelme_to_yolo(
    json_dir,
    image_dir,
    output_label_dir,
    output_image_dir,
    train_dir,
    val_dir,
    test_dir,
    split_ratios=(0.8, 0.1, 0.1),
    bbox_expansion=25,
):
    """Convert Labelme line/point annotations directly to YOLO detection format.

    Rules:
    - All shapes with shape_type == 'linestrip' are used as objects
    - Labels with same corn_id (stem or leaf) are grouped together
    - Each group gets one bbox that encloses all its points
    - All objects are class 'plant' (class_id = 0)
    - Output format: class_id x_center y_center width height (no keypoints)
    """

    print("Checking input directories:")
    print(f"json_dir: {json_dir} (exists: {os.path.exists(json_dir)})")
    print(f"image_dir: {image_dir} (exists: {os.path.exists(image_dir) if image_dir else False})")
    print(f"output_dir: {output_image_dir} (will be created if not exists)")

    # Ensure output directories exist
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Created output directories: {train_dir}, {val_dir}, {test_dir}")

    # Get all JSON files
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {json_dir}")

    if not json_files:
        print("Error: No JSON files found in json_dir. Check the directory and file naming.")
        return [], [], []

    random.shuffle(json_files)  # Shuffle for random splitting

    # Calculate split indices
    total_files = len(json_files)
    train_count = int(total_files * split_ratios[0])
    val_count = int(total_files * split_ratios[1])
    test_count = total_files - train_count - val_count
    print(f"Total files: {total_files}, Train: {train_count}, Val: {val_count}, Test: {test_count}")

    # Split JSON files
    train_files = json_files[:train_count]
    val_files = json_files[train_count:train_count + val_count]
    test_files = json_files[train_count + val_count:]

    # Only one class: plant
    class_name = "plant"

    # Process each split
    for split_name, split_files, split_img_dir in [
        ("train", train_files, train_dir),
        ("val", val_files, val_dir),
        ("test", test_files, test_dir),
    ]:
        print(f"Processing {split_name} set with {len(split_files)} files...")
        for json_path in tqdm(split_files, desc=f"Processing {split_name} Labelme JSON"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "shapes" not in data:
                    print(f"Warning: No 'shapes' in {json_path}, skipping.")
                    continue

                img_width = data.get("imageWidth")
                img_height = data.get("imageHeight")

                # DJI_0111.json -> DJI_0111.JPG
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                img_name = base_name + ".JPG"

                # Prepare image output path in split directory
                os.makedirs(split_img_dir, exist_ok=True)
                split_img_path = os.path.join(split_img_dir, img_name)

                # 1) Try to copy from image_dir if provided and file exists
                copied_or_saved = False
                if image_dir:
                    candidate_paths = [
                        os.path.join(image_dir, img_name),
                        os.path.join(image_dir, base_name + ".jpg"),
                        os.path.join(image_dir, base_name + ".JPG"),
                    ]
                    for cand in candidate_paths:
                        if os.path.exists(cand):
                            shutil.copy(cand, split_img_path)
                            print(f"Copied image to {split_img_path}")
                            copied_or_saved = True
                            break

                # 2) If no file on disk, try to decode from embedded base64 in Labelme JSON
                if not copied_or_saved:
                    image_data = data.get("imageData")
                    if image_data:
                        copied_or_saved = _save_image_from_base64(image_data, split_img_path)

                if not copied_or_saved:
                    print(f"Warning: Could not find image file or embedded image data for {json_path}, skipping.")
                    continue

                # If width/height are missing, try to read them from saved image
                if img_width is None or img_height is None:
                    try:
                        with Image.open(split_img_path) as im:
                            img_width, img_height = im.size
                    except Exception as e:
                        print(f"Error: cannot get image size for {split_img_path}: {e}")
                        continue

                if not img_width or not img_height:
                    print(f"Warning: invalid image size for {json_path}, skipping.")
                    continue

                # --- Core modification: Group by corn_id and merge into one bbox ---
                # Collect all points for each corn_id
                plant_groups = {}  # corn_id -> list of points

                for shape in data.get("shapes", []):
                    if shape.get("shape_type") != "linestrip" or not shape.get("label"):
                        continue

                    _, corn_id = _parse_label(shape["label"])
                    points = shape.get("points", [])
                    if not points:
                        continue

                    if corn_id not in plant_groups:
                        plant_groups[corn_id] = []
                    plant_groups[corn_id].extend(points)

                if not plant_groups:
                    print(f"No valid linestrip annotations in {json_path}, skipping.")
                    continue

                # Create YOLO label file
                label_path = os.path.join(output_label_dir, split_name, os.path.splitext(img_name)[0] + ".txt")
                os.makedirs(os.path.dirname(label_path), exist_ok=True)

                lines_written = 0
                with open(label_path, "w", encoding="utf-8") as lf:
                    for corn_id, all_points in plant_groups.items():
                        # Compute bbox that encloses all points of this plant
                        bbox = _get_bbox_from_points(all_points, expansion=bbox_expansion)
                        x, y, w, h = bbox

                        # Normalize
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        w_norm = w / img_width
                        h_norm = h / img_height

                        # Fixed class: plant (class_id = 0)
                        class_id = 0

                        # YOLO format: class_id x_center y_center width height
                        line = [class_id, x_center, y_center, w_norm, h_norm]
                        lf.write(" ".join(map(str, line)) + "\n")
                        lines_written += 1

                if lines_written == 0:
                    # No valid annotations written, remove empty label file
                    os.remove(label_path)
                    print(f"No valid annotations in {json_path}, removed empty label {label_path}")
                else:
                    print(f"Created label file: {label_path} with {lines_written} plant(s)")

            except Exception as e:
                print(f"Error processing {json_path}: {str(e)}")
                continue

    return train_files, val_files, test_files


def create_yaml(output_dir, train_dir, val_dir, test_dir):
    """Create YOLO data.yaml with single class 'plant'."""
    names = ["plant"]
    yaml_content = f"""
train: {train_dir}/
val: {val_dir}/
test: {test_dir}/

nc: {len(names)}
names: {names}
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    print(f"Created data.yaml at {yaml_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Labelme linestrip annotations to YOLO detection format (one bbox per plant)."
    )
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing Labelme JSON files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for YOLO dataset.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help=(
            "Directory containing original images. If omitted or image file is missing, "
            "the script will try to decode base64 image data from Labelme JSON (imageData)."
        ),
    )
    parser.add_argument("--bbox_expansion", type=int, default=25, help="BBox expansion in pixels (default: 25)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)")
    return parser.parse_args()


def main():
    args = parse_args()

    json_dir = args.json_dir
    output_dir = args.output_dir
    image_dir = args.image_dir

    # Normalise split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if total_ratio <= 0:
        raise ValueError("Sum of train/val/test ratios must be > 0")
    split_ratios = (
        args.train_ratio / total_ratio,
        args.val_ratio / total_ratio,
        args.test_ratio / total_ratio,
    )

    output_label_dir = os.path.join(output_dir, "labels")
    output_image_dir = os.path.join(output_dir, "images")

    # Define split directories
    train_dir = os.path.join(output_image_dir, "train")
    val_dir = os.path.join(output_image_dir, "val")
    test_dir = os.path.join(output_image_dir, "test")

    # Convert Labelme to YOLO and split dataset
    train_files, val_files, test_files = convert_labelme_to_yolo(
        json_dir,
        image_dir,
        output_label_dir,
        output_image_dir,
        train_dir,
        val_dir,
        test_dir,
        split_ratios=split_ratios,
        bbox_expansion=args.bbox_expansion,
    )
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

    # Create data.yaml (single class: plant)
    create_yaml(output_dir, train_dir, val_dir, test_dir)


if __name__ == "__main__":
    main()