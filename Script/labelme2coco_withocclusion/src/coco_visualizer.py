import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from .utils import load_json, logger

def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Generate distinct BGR colors for categories."""
    try:
        cmap = plt.get_cmap('tab20', num_colors)
        return [(int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255)) for rgba in cmap(np.arange(num_colors))]
    except Exception:
        logger.warning("Using predefined color palette")
        return [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ][:num_colors]

def visualize_coco(coco_json_path: str, image_dir: str, output_dir: str, vis_config: dict) -> bool:
    try:
        logger.info(f"Visualizing COCO annotations: {coco_json_path}")
        coco_data = load_json(coco_json_path)

        # Prepare categories
        categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
        if not categories:
            logger.error("No categories found")
            return False

        # Assign colors
        cat_id_to_color = {cat_id: color for cat_id, color in zip(categories.keys(), get_colors(len(categories)))}

        # Process images
        images_info = {img['id']: img for img in coco_data.get('images', [])}
        if not images_info:
            logger.error("No images found")
            return False

        for image_id, image_info in images_info.items():
            image_path = os.path.join(image_dir, image_info['file_name'])
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                continue

            # Visualize annotations
            annotations = [ann for ann in coco_data.get('annotations', []) if ann['image_id'] == image_id]
            for ann in annotations:
                cat_id = ann.get('category_id')
                if cat_id not in categories:
                    logger.warning(f"Skipping annotation with unknown category_id {cat_id}")
                    continue

                cat_info = categories[cat_id]
                keypoints = np.array(ann.get('keypoints', [])).reshape(-1, 3)
                if len(keypoints) == 0:
                    logger.warning(f"Annotation ID {ann['id']} has no keypoints")
                    continue

                # Draw bounding box
                if 'bbox' in ann:
                    x, y, w, h = map(int, ann['bbox'])
                    cv2.rectangle(image, (x, y), (x + w, y + h), vis_config['bbox_color'], vis_config['bbox_thickness'])
                    
                    label_pos = (x, y - vis_config['text_offset'] if y - vis_config['text_offset'] > vis_config['text_offset'] else y + vis_config['text_offset'])
                    cv2.putText(image, f"{cat_info['name']} (corn_id: {ann.get('corn_id', 'N/A')})",
                                label_pos, cv2.FONT_HERSHEY_SIMPLEX, vis_config['font_scale'],
                                vis_config['text_color'], vis_config['font_thickness'], cv2.LINE_AA)

                # Draw keypoints
                for i, (x, y, v) in enumerate(keypoints):
                    x, y = int(x), int(y)
                    if v == 2:  # Visible
                        cv2.circle(image, (x, y), vis_config['keypoint_radius'], vis_config['normal_color'], -1)
                        cv2.putText(image, str(i+1), (x + vis_config['keypoint_label_offset'], y - vis_config['keypoint_label_offset']),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, vis_config['text_color'], 1)
                    elif v == 1:  # Occluded
                        cv2.circle(image, (x, y), vis_config['keypoint_occluded_radius'], vis_config['occluded_color'], -1)
                        cv2.circle(image, (x, y), vis_config['keypoint_occluded_radius'], (0, 0, 0), 1)
                        cv2.putText(image, str(i+1), (x + vis_config['keypoint_label_offset'], y - vis_config['keypoint_label_offset']),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, vis_config['text_color'], 1)

                # Draw skeleton
                for connection in cat_info.get('skeleton', []):
                    p1_idx, p2_idx = connection
                    if not (0 <= p1_idx < len(keypoints) and 0 <= p2_idx < len(keypoints)):
                        logger.warning(f"Invalid skeleton connection [{p1_idx},{p2_idx}]")
                        continue
                    pt1, pt2 = keypoints[p1_idx], keypoints[p2_idx]
                    if pt1[2] > 0 and pt2[2] > 0:
                        x1, y1, x2, y2 = map(int, [pt1[0], pt1[1], pt2[0], pt2[1]])
                        line_color = vis_config['occluded_color'] if pt1[2] == 1 or pt2[2] == 1 else vis_config['normal_color']
                        cv2.line(image, (x1, y1), (x2, y2), line_color, vis_config['line_thickness'])
                        if line_color == vis_config['occluded_color']:
                            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # Save visualization with proper filename
            base_name = os.path.splitext(image_info['file_name'])[0]
            output_image_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
            cv2.imwrite(output_image_path, image)
            logger.info(f"Saved visualization to: {output_image_path}")

        return True
    except Exception as e:
        logger.error(f"Failed to visualize {coco_json_path}: {e}")
        return False