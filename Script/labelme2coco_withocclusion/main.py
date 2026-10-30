import os
from src.utils import set_working_directory, load_json, logger
from src.preprocessor import json2jpg, sort_and_interpolate_points
from src.labelme_parser import add_visibility_to_labelme
from src.coco_converter import labelme_to_coco, convert_folder_to_coco
from src.coco_visualizer import visualize_coco

def main():
    set_working_directory()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = load_json(config_path)
    
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    bbox_expansion = config['bbox_expansion']
    occlusion_threshold = config['occlusion_threshold']
    vis_config = config['visualization']
    preprocess_config = config['preprocessing']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    
    temp_extract_dir = os.path.join(output_dir, 'temp_extracted')
    temp_process_dir = os.path.join(output_dir, 'temp_processed')


    # Step 1: Preprocess - Extract images from JSON files
    json2jpg(input_dir, temp_extract_dir, preprocess_config['ratio'])
    
    # Step 2: Preprocess - Sort and interpolate points
    sort_and_interpolate_points(temp_extract_dir, temp_process_dir, 
                              preprocess_config['num_points'], 
                              preprocess_config['num_pad'])
    
    # Step 3: Process JSON files for visibility, COCO conversion, and visualization
    processed_json_dir = os.path.join(output_dir, 'processed_jsons')
    os.makedirs(processed_json_dir, exist_ok=True)

    for filename in os.listdir(temp_process_dir):
        if not filename.endswith('.json'):
            continue
            
        input_path = os.path.join(temp_process_dir, filename)
        base_name = os.path.splitext(filename)[0]
        vis_json_path = os.path.join(processed_json_dir, f"{base_name}.json")
        image_path = os.path.join(output_dir, f"{base_name}.jpg")
        

        if not add_visibility_to_labelme(input_path, vis_json_path, image_path, occlusion_threshold, output_dir):
            continue

    # Step 3.2
    merged_coco_path = os.path.join(output_dir, 'annotations.json')
    if convert_folder_to_coco(processed_json_dir, merged_coco_path, bbox_expansion):
        logger.info(f"成功生成合并的COCO文件: {merged_coco_path}")

        # Step 3.3
        if vis_config.get('enable', True):
            vis_image_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_image_dir, exist_ok=True)
            visualize_coco(merged_coco_path, output_dir, vis_image_dir, vis_config)
        else:
            logger.info("配置中关闭了可视化生成，跳过 visualize_coco。")
    else:
        logger.error("批量转换COCO格式失败")

    for temp_dir in [temp_extract_dir, temp_process_dir, processed_json_dir]:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()