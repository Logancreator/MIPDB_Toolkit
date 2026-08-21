import os
import sys
import json
import base64
from io import BytesIO
from PIL import Image
import logging
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_script_dir() -> str:
    """Get the directory of the script or executable."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def set_working_directory() -> None:
    """Set the working directory to the script's directory."""
    try:
        dir_path = get_script_dir()
        os.chdir(dir_path)
        logger.info(f"Working directory set to: {dir_path}")
    except Exception as e:
        logger.warning(f"Could not set working directory: {e}")
        logger.info(f"Proceeding with current directory: {os.getcwd()}")

def load_json(file_path: str) -> dict:
    """Load a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved JSON to: {file_path}")
    except IOError as e:
        logger.error(f"Could not write JSON to {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving JSON to {file_path}: {e}")
        raise

def save_image_from_data(image_data: str, output_dir: str, filename: str) -> Optional[str]:
    """Save image from base64 data."""
    try:
        img_data = base64.b64decode(image_data)
        img_pil = Image.open(BytesIO(img_data))
        if img_pil.mode == 'RGBA':
            img_pil = img_pil.convert('RGB')
        output_path = os.path.join(output_dir, filename)
        img_pil.save(output_path)
        logger.info(f"Saved image to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def get_bbox_from_points(points: List[List[float]], expansion: int = 0) -> List[float]:
    """Calculate TLWH bbox from points with optional expansion."""
    if not points:
        return [0, 0, 0, 0]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, y_min = min(x_coords) - expansion, min(y_coords) - expansion
    x_max, y_max = max(x_coords) + expansion, max(y_coords) + expansion
    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)
    return [x_min, y_min, width, height]

def parse_label(label: str) -> Tuple[str, str]:
    """Parse LabelMe label to determine category and corn_id."""
    if '-y' in label:
        corn_id = label.split('-y')[0]
        return 'leaf', corn_id
    return 'stem', label