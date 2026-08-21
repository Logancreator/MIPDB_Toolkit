import base64
import json
import os
import random
from PIL import Image, ImageOps
from tqdm import tqdm
from .utils import logger

def json2jpg(jsons_dir, save_dir, ratio=1):
    """
    Extract image data from JSON files and save as JPG files, while keeping the original JSON files
    
    Parameters:
    jsons_dir: JSON files directory
    save_dir: Directory to save images and JSON files
    ratio: Image scaling ratio
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_files = [f for f in os.listdir(jsons_dir) if f.endswith('.json')]
    logger.info(f"Total JSON files: {len(total_files)}")

    for jsonFile in tqdm(total_files, desc="Extracting images"):
        try:
            jsonPath = os.path.join(jsons_dir, jsonFile)
            with open(jsonPath, 'r', encoding='UTF-8') as f:
                data = json.load(f)

                # Check if imageData exists in the JSON file
                if 'imageData' not in data or data['imageData'] is None:
                    logger.warning(f"No imageData found in {jsonFile}, skipping...")
                    continue

                base_name = jsonFile.split('.')[0]
                image_name = base_name + ".JPG"
                jpgPath = os.path.join(save_dir, image_name)
                json_save_path = os.path.join(save_dir, jsonFile)

                # Skip if both files already exist
                if os.path.exists(jpgPath) and os.path.exists(json_save_path):
                    logger.info(f"Files for {base_name} already exist, skipping...")
                    continue

                # Safely decode image data
                try:
                    imageData = base64.b64decode(data['imageData'])
                except Exception as e:
                    logger.error(f"Failed to decode image data in {jsonFile}: {e}")
                    continue

                # Write image file
                with open(jpgPath, "wb") as file:
                    file.write(imageData)

                # Process image
                try:
                    image = Image.open(jpgPath)
                    image = ImageOps.exif_transpose(image)
                    if ratio != 1:
                        image = image.resize((int(image.size[0] / ratio), int(image.size[1] / ratio)))
                    image.save(jpgPath)

                    # Update image dimensions in JSON data
                    data['imageWidth'], data['imageHeight'] = image.size
                    
                    # Save JSON file
                    with open(json_save_path, 'w', encoding='utf-8') as json_file:
                        json.dump(data, json_file, ensure_ascii=False, indent=2)

                    # Get JSON file modification time
                    src_mtime = os.path.getmtime(jsonPath)
                    src_atime = os.path.getatime(jsonPath)

                    # Apply JSON file modification time to both new files
                    os.utime(jpgPath, (src_atime, src_mtime))
                    os.utime(json_save_path, (src_atime, src_mtime))
                except Exception as e:
                    logger.error(f"Error processing image {image_name}: {e}")
                    if os.path.exists(jpgPath):
                        os.remove(jpgPath)  # Clean up partial files
        except Exception as e:
            logger.error(f"Error processing {jsonFile}: {e}")

def sort_and_interpolate_points(json_dir, save_dir, num_selected_points=10, num_pad=10):
    """
    Sort and interpolate points in JSON files, keeping the Labelme format
    
    Parameters:
    json_dir: Directory containing JSON files
    save_dir: Directory to save processed JSON files
    num_selected_points: Final number of points to select
    num_pad: Number of additional points between each pair of points during interpolation
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    json_list = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    random.shuffle(json_list)

    # Define log file path
    log_file = os.path.join(save_dir, 'interpolation_warnings.log')

    for file in tqdm(json_list, desc="Processing points"):
        file_path = os.path.join(json_dir, file)
        new_file_path = os.path.join(save_dir, file)

        # Skip if file already exists in target path
        if os.path.exists(new_file_path):
            logger.info(f"Processed file {file} already exists, skipping...")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                for item in data['shapes']:
                    # Skip point type or label 'o' shapes to avoid interpolation
                    if item.get('shape_type') == 'point' or item['label'] == 'o':
                        continue  # Do not process occlusion points or point shapes

                    original_points_count = len(item['points'])  # Record original point count
                    if 'y' not in item['label']:
                        # Process root points
                        root = item['points']
                        sorted_root = sorted(root, key=lambda x: x[1], reverse=True)
                        item['points'] = sorted_root
                    else:
                        belong_root = None
                        belong_label = item['label'].split('-')[0]

                        for temp in data['shapes']:
                            if belong_label == temp['label']:
                                belong_root = temp['points']
                                break

                        if belong_root is not None:
                            mean_x = sum(point[0] for point in belong_root) / len(belong_root)
                            mean_y = sum(point[1] for point in belong_root) / len(belong_root)

                            length_first = (item['points'][0][0] - mean_x) ** 2 + (item['points'][0][1] - mean_y) ** 2
                            length_last = (item['points'][-1][0] - mean_x) ** 2 + (item['points'][-1][1] - mean_y) ** 2

                            if length_first > length_last:
                                item['points'] = item['points'][::-1]

                    # Interpolate to increase point count
                    new_points = []
                    if len(item['points']) < 2:
                        logger.warning(f"Warning: {file}'s {item['label']} has fewer than 2 points, cannot interpolate. Will use original points or default points.")
                        with open(log_file, 'a', encoding='utf-8') as log:
                            log.write(f"{file}, label: {item['label']}, original points: {original_points_count}, interpolated points: 0, target points: {num_selected_points}, note: fewer than 2 points\n")

                        if item['points']:
                            new_points = item['points'] * num_selected_points
                        else:
                            new_points = [[0, 0]] * num_selected_points
                    else:
                        for i in range(len(item['points']) - 1):
                            x1, y1 = item['points'][i]
                            x2, y2 = item['points'][i + 1]

                            for j in range(num_pad + 1):
                                x = x1 + (x2 - x1) * j / (num_pad + 1)
                                y = y1 + (y2 - y1) * j / (num_pad + 1)
                                new_points.append([x, y])

                            if i == len(item['points']) - 2:
                                new_points.append([x2, y2])

                    if len(new_points) < num_selected_points:
                        logger.warning(f"Warning: {file}'s {item['label']} has fewer than {num_selected_points} interpolated points, currently {len(new_points)}. Will repeat the last point to fill.")
                        with open(log_file, 'a', encoding='utf-8') as log:
                            log.write(f"{file}, label: {item['label']}, original points: {original_points_count}, interpolated points: {len(new_points)}, target points: {num_selected_points}, note: insufficient interpolated points\n")
                        if not new_points:
                            new_points = [[0, 0]] * num_selected_points
                        else:
                            while len(new_points) < num_selected_points:
                                new_points.append(new_points[-1])

                    num_points = len(new_points)
                    step = (num_points - 1) / (num_selected_points - 1) if num_selected_points > 1 else 0

                    selected_points = []
                    for i in range(num_selected_points):
                        index = round(i * step)
                        if index >= len(new_points):
                            index = len(new_points) - 1
                        selected_points.append(new_points[index])

                    if len(selected_points) != num_selected_points:
                        logger.warning(f"Warning: {file}'s {item['label']} has {len(selected_points)} target points, should be {num_selected_points}")
                        with open(log_file, 'a', encoding='utf-8') as log:
                            log.write(f"{file}, label: {item['label']}, original points: {original_points_count}, interpolated points: {len(new_points)}, target points: {len(selected_points)}, note: incorrect target points\n")

                    item['points'] = selected_points

            # Save new JSON file
            with open(new_file_path, 'w', encoding='utf-8') as new_f:
                json.dump(data, new_f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")