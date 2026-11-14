import json
import os
import glob
import shutil
import argparse
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from point_resampling import resample_polyline, resample_visibility
import random


def _save_image_from_base64(image_data, output_path):
    """Decode base64 image data and save to output_path.

    Returns (success, image_width, image_height).
    """
    try:
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"Saved image to {output_path} from embedded base64 data")
        return True, img.width, img.height
    except Exception as e:
        print(f"Error decoding/saving embedded image to {output_path}: {e}")
        return False, None, None


def _parse_label(label):
    """Parse Labelme label, consistent with labelme2coco_script.parse_label.

    Rules:
    - Labels containing '-y' are treated as leaves ("leaf");
    - Other labels are treated as stems ("stem").
    Returns (category_name, corn_id).
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
    num_keypoints=17,
    resize_scale=1.0,  # 1.0 = original, 2.0 = half size, 0.5 = double size
    interpolation_method="legacy",
):
    """Convert Labelme line/point annotations directly to YOLO keypoint format.

    Rules:
    - Only shapes with shape_type == 'linestrip' are used as objects
        (point labels like 'o' or 'q' are ignored).
    - Labels without '-y' are treated as class 'stem', labels with '-y'
        are treated as class 'leaf'.
    - Each linestrip is one object; its points become keypoints. Visibility:
        1 -> visible (v=2), 0 -> occluded (v=1).
    """

    print("Checking input directories:")
    print(f"json_dir: {json_dir} (exists: {os.path.exists(json_dir)})")
    print(f"image_dir: {image_dir} (exists: {os.path.exists(image_dir) if image_dir else False})")
    print(f"output_dir: {output_image_dir} (will be created if not exists)")
    if resize_scale != 1.0:
        print(f"Resize scale: 1/{resize_scale:.2f} (new size = original / {resize_scale:.2f})")
    else:
        print("No resize (scale = 1.0)")

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

    # Classes are fixed as: stem, leaf
    category_names = ["stem", "leaf"]

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

                # DJI_0111.json -> DJI_0111.JPG、DJI_0111.txt
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                img_name = base_name + ".JPG"

                # Prepare image output path in split directory
                os.makedirs(split_img_dir, exist_ok=True)
                split_img_path = os.path.join(split_img_dir, img_name)

                # Load image for resizing
                img = None
                copied_or_saved = False
                original_width = img_width
                original_height = img_height

                # 1) Try to copy from image_dir if provided and file exists
                if image_dir:
                    candidate_paths = [
                        os.path.join(image_dir, img_name),
                        os.path.join(image_dir, base_name + ".jpg"),
                        os.path.join(image_dir, base_name + ".JPG"),
                        os.path.join(image_dir, base_name + ".jpeg"),
                        os.path.join(image_dir, base_name + ".JPEG"),
                        os.path.join(image_dir, base_name + ".png"),
                        os.path.join(image_dir, base_name + ".PNG"),
                    ]
                    for cand in candidate_paths:
                        if os.path.exists(cand):
                            try:
                                img = Image.open(cand)
                                if img.mode == "RGBA":
                                    img = img.convert("RGB")
                                original_width, original_height = img.size
                                copied_or_saved = True
                                print(f"Loaded image from {cand}")
                                break
                            except Exception as e:
                                print(f"Error loading image {cand}: {e}")
                                continue

                # 2) If no file on disk, try to decode from embedded base64 in Labelme JSON
                if not copied_or_saved:
                    image_data = data.get("imageData")
                    if image_data:
                        try:
                            img_bytes = base64.b64decode(image_data)
                            img = Image.open(BytesIO(img_bytes))
                            if img.mode == "RGBA":
                                img = img.convert("RGB")
                            original_width, original_height = img.size
                            copied_or_saved = True
                            print(f"Decoded image from base64 data")
                        except Exception as e:
                            print(f"Error decoding base64 image: {e}")

                if not copied_or_saved or img is None:
                    print(f"Warning: Could not find image file or embedded image data for {json_path}, skipping.")
                    continue

                # If width/height are missing from JSON, use loaded image
                if img_width is None or img_height is None:
                    img_width, img_height = original_width, original_height

                if not img_width or not img_height:
                    print(f"Warning: invalid image size for {json_path}, skipping.")
                    continue

                # --- Resize image and update coordinates ---
                # First, collect all points from all shapes to update them
                shapes_updated = []
                for shape in data.get("shapes", []):
                    if shape.get("shape_type") != "linestrip" or not shape.get("label"):
                        continue
                    points = shape.get("points", [])
                    if not points:
                        continue
                    # Copy shape and update points
                    new_shape = shape.copy()
                    new_shape["points"] = points.copy()
                    shapes_updated.append(new_shape)

                # Apply resize if scale != 1.0
                if resize_scale != 1.0 and shapes_updated:
                    # Calculate scaling factor (new_size = original / resize_scale)
                    scale_factor = 1.0 / resize_scale
                    target_width = int(img_width * scale_factor)
                    target_height = int(img_height * scale_factor)
                    
                    print(f"Resizing from {img_width}x{img_height} to {target_width}x{target_height} (scale: {scale_factor:.4f})")
                    
                    # Scale all points in all shapes
                    for shape in shapes_updated:
                        points = shape["points"]
                        scaled_points = []
                        for x, y in points:
                            scaled_points.append([x * scale_factor, y * scale_factor])
                        shape["points"] = scaled_points
                    
                    # Resize image
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    img_width, img_height = target_width, target_height

                # Save resized image
                try:
                    img.save(split_img_path)
                    print(f"Saved image to {split_img_path}")
                except Exception as e:
                    print(f"Error saving image {split_img_path}: {e}")
                    continue

                # Create YOLO label file
                label_subdir = os.path.join(output_label_dir, split_name)
                os.makedirs(label_subdir, exist_ok=True)
                label_path = os.path.join(label_subdir, os.path.splitext(img_name)[0] + ".txt")

                lines_written = 0
                with open(label_path, "w", encoding="utf-8") as lf:
                    for shape in shapes_updated:
                        category_name, _ = _parse_label(shape["label"])
                        if category_name not in category_names:
                            continue

                        class_id = category_names.index(category_name)
                        points = shape.get("points", [])
                        if not points:
                            continue

                        visibility = shape.get("visibility", [1] * len(points))
                        source_count = len(points)
                        points = resample_polyline(
                            points, num_keypoints, interpolation_method
                        )
                        visibility = resample_visibility(
                            visibility, source_count, num_keypoints
                        )

                        # Enforce fixed number of keypoints: truncate or pad with the last point
                        if len(points) < num_keypoints:
                            if not points:
                                points = [[0.0, 0.0]] * num_keypoints
                                visibility = [0] * num_keypoints
                            else:
                                while len(points) < num_keypoints:
                                    points.append(points[-1])
                                while len(visibility) < num_keypoints:
                                    visibility.append(visibility[-1])
                        elif len(points) > num_keypoints:
                            points = points[:num_keypoints]
                            visibility = visibility[:num_keypoints]

                        # Compute bbox (using updated points)
                        bbox = _get_bbox_from_points(points, expansion=bbox_expansion)
                        x, y, w, h = bbox
                        # 归一化并确保不超过1
                        x_center = min((x + w / 2) / img_width, 1.0)
                        y_center = min((y + h / 2) / img_height, 1.0)
                        w_norm = min(w / img_width, 1.0)
                        h_norm = min(h / img_height, 1.0)

                        # Normalize keypoints and encode visibility (same convention as COCO)
                        keypoints_norm = []
                        for i in range(num_keypoints):
                            kx, ky = points[i]
                            lab_v = visibility[i] if i < len(visibility) else 1
                            # ===== 边界检查和裁剪（新增） =====
                            # 检查关键点是否超出图像边界
                            is_outside = False
                            if kx < 0:
                                kx = 0
                                is_outside = True
                            elif kx > img_width:
                                kx = img_width
                                is_outside = True
                            
                            if ky < 0:
                                ky = 0
                                is_outside = True
                            elif ky > img_height:
                                ky = img_height
                                is_outside = True
                            
                            # 如果关键点超出边界，标记为不可见 (v=1)
                            if is_outside:
                                v = 1  # occluded
                            else:
                                # 根据原始visibility设置
                                if lab_v == 1:
                                    v = 2  # visible
                                elif lab_v == 0:
                                    v = 1  # occluded
                                else:
                                    v = 2
                            # ===== 边界检查和裁剪结束 =====
                            
                            # 归一化并确保不超过1
                            kx_norm = min(kx / img_width, 1.0)
                            ky_norm = min(ky / img_height, 1.0)
                            if abs(kx_norm - 1.0) < 1e-6:
                                kx_norm = 1.0
                            if abs(ky_norm - 1.0) < 1e-6:
                                ky_norm = 1.0
                            keypoints_norm.extend([kx_norm, ky_norm, v])

                        line = [class_id, x_center, y_center, w_norm, h_norm] + keypoints_norm
                        lf.write(" ".join(map(str, line)) + "\n")
                        lines_written += 1

                if lines_written == 0:
                    # No valid annotations written, remove empty label file
                    if os.path.exists(label_path):
                        os.remove(label_path)
                    print(f"No valid linestrip annotations in {json_path}, removed empty label {label_path}")
                else:
                    print(f"Created label file: {label_path} with {lines_written} objects")

            except Exception as e:
                print(f"Error processing {json_path}: {str(e)}")
                continue

    return train_files, val_files, test_files


def create_yaml(output_dir, num_keypoints, train_dir, val_dir, test_dir):
    """Create YOLO data.yaml with fixed classes 'stem' and 'leaf'."""
    names = ["stem", "leaf"]
    yaml_content = f"""train: {train_dir}/
val: {val_dir}/
test: {test_dir}/

nc: {len(names)}
names: {names}

kpt_shape: [{num_keypoints}, 3]
flip_idx: [{", ".join(str(i) for i in range(num_keypoints))}]
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    print(f"Created data.yaml at {yaml_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Labelme linestrip keypoints to YOLO format.")
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
    parser.add_argument("--num_keypoints", type=int, default=17, help="Number of keypoints per instance (default: 17)")
    parser.add_argument(
        "--interpolation_method",
        choices=["legacy", "arc_length"],
        default="arc_length",
        help="Point resampling method (default: arc_length)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)")
    parser.add_argument(
        "--resize",
        type=float,
        default=4,
        help="Resize scale: 1=original, 2=half size (width/2, height/2), 0.5=double size. Default: 1.0"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    json_dir = args.json_dir
    output_dir = args.output_dir
    image_dir = args.image_dir

    # Validate resize scale
    if args.resize <= 0:
        raise ValueError(f"Resize scale must be positive, got {args.resize}")

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
        num_keypoints=args.num_keypoints,
        resize_scale=args.resize,
        interpolation_method=args.interpolation_method,
    )
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

    # Create data.yaml (classes fixed as stem/leaf)
    create_yaml(output_dir, args.num_keypoints, train_dir, val_dir, test_dir)


if __name__ == "__main__":
    main()