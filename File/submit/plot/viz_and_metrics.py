import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def load_yolo_txt(file_path, img_w, img_h, num_kpts=17):
    if not os.path.exists(file_path):
        return [], []
    
    bboxes = []
    keypoints = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5:
                continue
            
            cx, cy, w, h = parts[1:5]
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            bboxes.append([x1, y1, x2, y2])
            
            kpt_parts = parts[5:]
            kpts = []
            vals_per_kpt = len(kpt_parts) // num_kpts
            for i in range(num_kpts):
                idx = i * vals_per_kpt
                px = kpt_parts[idx] * img_w
                py = kpt_parts[idx+1] * img_h
                pv = kpt_parts[idx+2] if vals_per_kpt == 3 else 1.0
                kpts.append([px, py, pv])
            keypoints.append(kpts)
            
    return np.array(bboxes), np.array(keypoints)

def calculate_oks(gt, pred, area, sigmas):
    visible = gt[:, 2] > 0
    if not np.any(visible):
        return 0.0
    
    dx = gt[:, 0] - pred[:, 0]
    dy = gt[:, 1] - pred[:, 1]
    dist_sq = dx**2 + dy**2
    
    kappa = 2 * (sigmas**2)
    oks_vals = np.exp(-dist_sq / (area * kappa + 1e-9))
    
    return np.sum(oks_vals[visible]) / np.sum(visible)

def calculate_pck(gt, pred, threshold):
    visible = gt[:, 2] > 0
    if not np.any(visible):
        return 0.0
    
    dx = gt[:, 0] - pred[:, 0]
    dy = gt[:, 1] - pred[:, 1]
    dist = np.sqrt(dx**2 + dy**2)
    
    correct = dist[visible] < threshold
    return np.mean(correct)

def greedy_match(gt_kpts, gt_bboxes, pred_kpts, sigmas):
    if len(gt_kpts) == 0 or len(pred_kpts) == 0:
        return []
    
    matches = []
    available_preds = set(range(len(pred_kpts)))
    
    for i in range(len(gt_kpts)):
        best_oks = -1
        best_idx = -1
        
        b = gt_bboxes[i]
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area <= 0: area = 1e-6
        
        for j in available_preds:
            oks = calculate_oks(gt_kpts[i], pred_kpts[j], area, sigmas)
            if oks > best_oks:
                best_oks = oks
                best_idx = j
        
        if best_idx != -1 and best_oks > 0.1:
            matches.append((i, best_idx))
            available_preds.remove(best_idx)
            
    return matches

def visualize_results(img, gt_kpts, pred_kpts, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    # Adaptive point size and line thickness based on image size
    thickness = max(2, int(max(h, w) / 600)) # e.g., 20 for 4000px
    radius = max(3, int(max(h, w) / 500))    # e.g., 26 for 4000px
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    img_gt = img_rgb.copy()
    for obj_kpts in gt_kpts:
        # Draw connections
        for i in range(len(obj_kpts) - 1):
            kp1, kp2 = obj_kpts[i], obj_kpts[i+1]
            # Filter low confidence and (0,0) coordinates
            if (kp1[2] > 0 and kp2[2] > 0 and 
                kp1[0] > 0 and kp1[1] > 0 and 
                kp2[0] > 0 and kp2[1] > 0):
                cv2.line(img_gt, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), (0, 255, 0), thickness)
        # Draw points
        for kp in obj_kpts:
            if kp[2] > 0 and kp[0] > 0 and kp[1] > 0:
                cv2.circle(img_gt, (int(kp[0]), int(kp[1])), radius, (0, 255, 0), -1)
    axes[1].imshow(img_gt)
    axes[1].set_title('Original + GT')
    axes[1].axis('off')
    
    img_pred = img_rgb.copy()
    for obj_kpts in pred_kpts:
        # Draw connections
        for i in range(len(obj_kpts) - 1):
            kp1, kp2 = obj_kpts[i], obj_kpts[i+1]
            # Filter low confidence and (0,0) coordinates
            if (kp1[2] > 0.3 and kp2[2] > 0.3 and 
                kp1[0] > 0 and kp1[1] > 0 and 
                kp2[0] > 0 and kp2[1] > 0):
                cv2.line(img_pred, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), (255, 0, 0), thickness)
        # Draw points
        for kp in obj_kpts:
            if kp[2] > 0.3 and kp[0] > 0 and kp[1] > 0:
                cv2.circle(img_pred, (int(kp[0]), int(kp[1])), radius, (255, 0, 0), -1)
    axes[2].imshow(img_pred)
    axes[2].set_title('Original + Pred')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def process_and_evaluate(img_dir, gt_label_dir, pred_label_dir, output_viz_dir, name):
    print(f'\n--- Evaluating: {name} ---')
    img_dir = Path(img_dir)
    gt_label_dir = Path(gt_label_dir)
    pred_label_dir = Path(pred_label_dir)
    output_viz_dir = Path(output_viz_dir) / name
    output_viz_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = list(img_dir.glob('*.JPG')) + list(img_dir.glob('*.PNG'))
    if not img_files:
        print(f'No images found in {img_dir}')
        return
    
    sigmas = np.array([0.1] * 17)
    all_oks = []
    all_pck = []
    pck_threshold_ratio = 0.05
    
    count = 0
    for img_path in tqdm(img_files, desc=f'Processing {name}'):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        gt_path = gt_label_dir / f'{img_path.stem}.txt'
        pred_path = pred_label_dir / f'{img_path.stem}.txt'
        
        gt_bboxes, gt_kpts = load_yolo_txt(gt_path, w, h)
        pred_bboxes, pred_kpts = load_yolo_txt(pred_path, w, h)
        
        pck_thresh = pck_threshold_ratio * max(w, h)
        matches = greedy_match(gt_kpts, gt_bboxes, pred_kpts, sigmas)
        
        for g_idx, p_idx in matches:
            b = gt_bboxes[g_idx]
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area <= 0: area = 1e-6
            oks = calculate_oks(gt_kpts[g_idx], pred_kpts[p_idx], area, sigmas)
            pck = calculate_pck(gt_kpts[g_idx], pred_kpts[p_idx], pck_thresh)
            all_oks.append(oks)
            all_pck.append(pck)
            
        if count < 20:
            visualize_results(img, gt_kpts, pred_kpts, output_viz_dir / f'{img_path.stem}_viz.png')
            count += 1
            
    if all_oks:
        avg_oks, avg_pck = np.mean(all_oks), np.mean(all_pck)
        print(f'[{name}] Results: Average OKS: {avg_oks:.4f}, Average PCK@{pck_threshold_ratio}: {avg_pck:.4f}')
        return avg_oks, avg_pck
    else:
        print(f'[{name}] No matches found.')

def main():
    eval_tasks = [
        {
            'name': 'UAV_Unpretrained',
            'img_dir': r'/public/home/panyuchen/data/corn/02.uav_17points_yolo/images/val',
            'gt_dir': r'/public/home/panyuchen/data/corn/02.uav_17points_yolo/labels/val',
            'pred_dir': r'PFLO_uav_unpretrained_div1/predict/labels',
            'out_viz': r'visualization_results'
        },
        {
            'name': 'DSLR_Pretrained',
            'img_dir': r'/public/home/panyuchen/data/corn/02.dslr_17points_yolo/images/val',
            'gt_dir': r'/public/home/panyuchen/data/corn/02.dslr_17points_yolo/labels/val',
            'pred_dir': r'PFLO_dslr_pretrained_div1/predict/labels',
            'out_viz': r'visualization_results'
        }
    ]
    for task in eval_tasks:
        process_and_evaluate(task['img_dir'], task['gt_dir'], task['pred_dir'], task['out_viz'], task['name'])

if __name__ == '__main__':
    main()