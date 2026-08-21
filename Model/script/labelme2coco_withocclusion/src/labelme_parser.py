import os
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict
from .utils import load_json, save_json, save_image_from_data, logger

def add_visibility_to_labelme(input_path: str, output_path: str, image_output_path: str, threshold: int, output_dir: str) -> bool:
    """Add visibility annotations to a LabelMe JSON based on occlusion points ('o') and correction points ('q')."""
    try:
        logger.info(f"Processing LabelMe JSON: {input_path}")
        data = load_json(input_path)

        # Save image if imageData exists
        if 'imageData' in data and data['imageData']:
            image_path = save_image_from_data(data['imageData'], output_dir, os.path.basename(image_output_path))
            if image_path:
                data['imagePath'] = os.path.basename(image_path)
            else:
                logger.warning(f"Failed to save image for {input_path}")
                return False
        else:
            logger.warning(f"No imageData found in {input_path}, using existing imagePath")

        # Collect linestrips, occlusion points ('o'), and correction points ('q')
        keypoints_by_linestrip: Dict[int, dict] = {}
        # We collect points as Python lists first, then convert to numpy arrays below.
        occlude_points = []
        correction_points = []
        
        for i, shape in enumerate(data['shapes']):
            if shape['shape_type'] == 'linestrip':
                keypoints_by_linestrip[i] = {
                    'points': shape['points'],
                    'label': shape['label'],
                    'visibility': [1] * len(shape['points'])
                }
            elif shape['label'] == 'o':
                occlude_points.append(shape['points'][0])
            elif shape['label'] == 'q':
                correction_points.append(shape['points'][0])

        occlude_points = np.array(occlude_points)
        correction_points = np.array(correction_points)

        # Assign visibility based on occlusion points ('o')
        occlusion_mapping = {}  # Track which keypoint was occluded by which 'o' point
        for i, linestrip in keypoints_by_linestrip.items():
            keypoints = np.array(linestrip['points'])
            if len(occlude_points) > 0:
                distances = cdist(keypoints, occlude_points)  # Shape: (num_keypoints, num_occlude_points)
                min_distances = distances.min(axis=1)  # Closest 'o' point distance for each keypoint
                min_occlude_indices = distances.argmin(axis=1)  # Index of closest 'o' point for each keypoint
                for k, (dist, occlude_idx) in enumerate(zip(min_distances, min_occlude_indices)):
                    if dist < threshold:
                        linestrip['visibility'][k] = 0
                        occlusion_mapping[occlude_idx] = (i, k)  # Store (linestrip_idx, keypoint_idx)
                        logger.info(f"Marked keypoint {k} in linestrip {i} as occluded by 'o' point {occlude_idx}")

        # Correct visibility based on correction points ('q')
        if len(correction_points) > 0 and len(occlude_points) > 0:
            distances = cdist(correction_points, occlude_points)
            for q_idx, min_occlude_idx in enumerate(distances.argmin(axis=1)):
                if min_occlude_idx in occlusion_mapping:
                    linestrip_idx, keypoint_idx = occlusion_mapping[min_occlude_idx]
                    keypoints_by_linestrip[linestrip_idx]['visibility'][keypoint_idx] = 1
                    logger.info(f"Corrected visibility for keypoint {keypoint_idx} in linestrip {linestrip_idx} due to 'q' point {q_idx}")

        # Update shapes with visibility
        for i, shape in enumerate(data['shapes']):
            if shape['shape_type'] == 'linestrip' and i in keypoints_by_linestrip:
                shape['visibility'] = keypoints_by_linestrip[i]['visibility']

        # Remove occlusion and correction points
        data['shapes'] = [shape for shape in data['shapes'] if shape['label'] not in ['o', 'q']]
        
        save_json(data, output_path)
        return True
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return False