import datetime
import os
from typing import List, Tuple, Dict
from .utils import load_json, save_json, get_bbox_from_points, logger, parse_label


def generate_skeleton(num_keypoints: int) -> List[List[int]]:
    """Generate skeleton connections for keypoints (0-based indexing)."""
    if num_keypoints <= 1:
        return []
    return [[i, i + 1] for i in range(num_keypoints - 1)]


def get_default_categories(num_keypoints: int = 17) -> Tuple[List[dict], Dict[str, dict]]:
    """Return default categories list and a lookup dict for category info."""
    keypoint_names = [f"kpt_{i+1}" for i in range(num_keypoints)]
    skeleton = generate_skeleton(num_keypoints)
    
    categories = [
        {"id": 1, "name": "stem", "supercategory": "corn", "keypoints": keypoint_names, "skeleton": skeleton},
        {"id": 2, "name": "leaf", "supercategory": "corn", "keypoints": keypoint_names, "skeleton": skeleton},
    ]

    lookup = {
        "stem": {"id": 1, "num_keypoints": num_keypoints},
        "leaf": {"id": 2, "num_keypoints": num_keypoints},
    }
    return categories, lookup


def labelme_to_coco_single(labelme_data: dict, bbox_expansion: int, image_id: int, annotation_start_id: int, category_lookup: Dict[str, dict]) -> Tuple[dict, List[dict], int]:
    """Convert one LabelMe JSON data (already loaded) into COCO image dict and list of annotations.

    Returns (image_info, annotations_list, next_annotation_id)
    """
    image_path = labelme_data.get("imagePath", f"unknown_{image_id}.jpg")
    image_info = {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "height": labelme_data.get("imageHeight", 0),
        "width": labelme_data.get("imageWidth", 0),
        "license": 1,
        "date_captured": ""
    }

    annotations = []
    ann_id = annotation_start_id

    for shape in labelme_data.get("shapes", []):
        if shape.get("shape_type") != "linestrip" or not shape.get("label"):
            logger.debug(f"Skipping shape with type '{shape.get('shape_type')}' or missing label")
            continue

        category_name, corn_id = parse_label(shape["label"])  # returns ('leaf'|'stem', corn_id)
        if category_name not in ["stem", "leaf"]:
            logger.warning(f"Skipping unknown category '{category_name}' for label '{shape.get('label')}')")
            continue

        points = shape.get("points", [])
        visibility = shape.get("visibility", [1] * len(points))

        if not points:
            logger.warning(f"Skipping annotation for label '{shape.get('label')}' due to empty points")
            continue

        # Use simple categories 'stem' and 'leaf'. Occlusion encoded in keypoint v (1=occluded, 2=visible)
        new_category_name = category_name
        if new_category_name not in category_lookup:
            logger.warning(f"Skipping unknown category '{new_category_name}' for label '{shape.get('label')}'")
            continue
        category_info = category_lookup[new_category_name]

        bbox = get_bbox_from_points(points, bbox_expansion)
        area = bbox[2] * bbox[3]

        # Format keypoints: map visibility explicitly (COCO v: 0=not labeled,1=labeled but not visible,2=labeled and visible)
        keypoints_coco = []
        num_labeled_keypoints = 0
        for i in range(category_info['num_keypoints']):
            if i < len(points):
                x, y = points[i]
                # Map LabelMe visibility: 1 -> visible (COCO v=2), 0 -> occluded (COCO v=1)
                if i >= len(visibility):
                    v = 2  # assume visible if no visibility info
                else:
                    lab_v = visibility[i]
                    if lab_v == 1:
                        v = 2
                    elif lab_v == 0:
                        v = 1
                    else:
                        v = 2

                keypoints_coco.extend([round(float(x), 2), round(float(y), 2), int(v)])
                if v > 0:
                    num_labeled_keypoints += 1
            else:
                keypoints_coco.extend([0, 0, 0])

        ann = {
            "id": ann_id,
            "image_id": image_info["id"],
            "category_id": category_info["id"],
            "corn_id": corn_id,
            "segmentation": [],
            "area": round(area, 2),
            "bbox": [round(coord, 2) for coord in bbox],
            "iscrowd": 0,
            "keypoints": keypoints_coco,
            "num_keypoints": num_labeled_keypoints,
            # "keypoint_labels": keypoint_labels,
        }

        annotations.append(ann)
        ann_id += 1

    return image_info, annotations, ann_id


def convert_folder_to_coco(input_dir: str, output_path: str, bbox_expansion: int = 0, num_keypoints: int = 17) -> bool:
    """Convert a folder of LabelMe JSON files into a single merged COCO JSON.

    - Scans input_dir for .json files, converts each, and merges into one COCO file.
    - Ensures unique image_id and annotation id across the dataset.
    """
    try:
        logger.info(f"Converting folder to merged COCO: {input_dir} -> {output_path}")

        coco_output = {
            "info": {
                "description": "Corn plant dataset for computer vision tasks",
                "url": "http://phenomics.agis.org.cn/#/category",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "Jianye Chang",
                "date_created": datetime.datetime.now().strftime("%Y/%m/%d")
            },
            "licenses": [{"url": "", "id": 1, "name": "Default License"}],
            "images": [],
            "annotations": [],
            "categories": []
        }

        categories, lookup = get_default_categories(num_keypoints)
        coco_output['categories'] = categories

        image_id = 1
        ann_id = 1

        files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        files.sort()
        for fname in files:
            path = os.path.join(input_dir, fname)
            try:
                data = load_json(path)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                continue

            img_info, anns, next_ann_id = labelme_to_coco_single(data, bbox_expansion, image_id, ann_id, lookup)
            # Only append image if there is at least one annotation
            if anns:
                coco_output['images'].append(img_info)
                coco_output['annotations'].extend(anns)
                image_id += 1
                ann_id = next_ann_id
            else:
                logger.info(f"No valid annotations found in {fname}; skipping adding image")

        save_json(coco_output, output_path)
        logger.info(f"Saved merged COCO to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to convert folder to COCO: {e}")
        return False
    

def labelme_to_coco(input_path: str, output_path: str, bbox_expansion: int = 0, num_keypoints: int = 17) -> bool:
    """Convert a single LabelMe JSON file to COCO format."""
    try:
        logger.info(f"Converting single LabelMe to COCO: {input_path} -> {output_path}")
        
        # Load LabelMe data
        labelme_data = load_json(input_path)
        
        # Get categories
        categories, lookup = get_default_categories(num_keypoints)
        
        # Convert single file
        image_info, annotations, _ = labelme_to_coco_single(
            labelme_data, bbox_expansion, image_id=1, annotation_start_id=1, category_lookup=lookup
        )
        
        # Create COCO format
        coco_output = {
            "info": {
                "description": "Corn plant dataset for computer vision tasks",
                "url": "http://phenomics.agis.org.cn/#/category",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "Jianye Chang",
                "date_created": datetime.datetime.now().strftime("%Y/%m/%d")
            },
            "licenses": [{"url": "", "id": 1, "name": "Default License"}],
            "images": [image_info] if annotations else [],
            "annotations": annotations,
            "categories": categories
        }
        
        save_json(coco_output, output_path)
        logger.info(f"Saved COCO to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {input_path} to COCO: {e}")
        return False