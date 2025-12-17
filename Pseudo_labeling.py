import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import torch

def xywh2xyxy(x, y, w, h):
    """Converts YOLO format (x_center, y_center, w, h) to corner coordinates (x1, y1, x2, y2)."""
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) ratio between two boxes."""
    # 
    # box: [class_id, x, y, w, h]
    
    # Corners of 1st box
    b1_x1, b1_y1, b1_x2, b1_y2 = xywh2xyxy(box1[1], box1[2], box1[3], box1[4])
    # Corners of 2nd box
    b2_x1, b2_y1, b2_x2, b2_y2 = xywh2xyxy(box2[1], box2[2], box2[3], box2[4])

    # Find intersection area
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Find union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area == 0: return 0
    return inter_area / union_area

def main(model_path, dataset_path, conf_threshold=0.5, iou_limit=0.45):
    """
    model_path: Path of the model to be used (.pt)
    dataset_path: Folder containing images (e.g., .../train/images)
    conf_threshold: Model confidence threshold (Don't write below 0.5)
    iou_limit: Do not write if overlaps more than 45% with existing label (Prevent duplicates)
    """
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Find images in dataset
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(valid_extensions)]
    
    print(f"Total {len(image_files)} images will be processed...")
    
    # Determine labels folder (.../train/images -> .../train/labels)
    # Usually the labels folder is a sibling of the images folder.
    parent_dir = os.path.dirname(dataset_path)
    labels_dir = os.path.join(parent_dir, "labels")
    
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
        print(f"Labels folder created: {labels_dir}")

    count_new_labels = 0

    # 
    for img_file in tqdm(image_files, desc="Labeling"):
        img_path = os.path.join(dataset_path, img_file)
        txt_name = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_name)

        # 1. READ EXISTING LABELS (If any)
        existing_boxes = []
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        existing_boxes.append(parts) # [class, x, y, w, h]

        # 2. MAKE MODEL PREDICTION
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        
        new_boxes_to_write = []
        
        for result in results:
            for box in result.boxes:
                # Model data
                cls_id = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist() # Normalized coordinates
                
                new_box = [cls_id, x, y, w, h]
                
                # 3. DUPLICATE CHECK (Compare with existing labels)
                is_duplicate = False
                for ex_box in existing_boxes:
                    # Only if same class and locations are very close (High IoU)
                    if int(ex_box[0]) == cls_id: 
                        iou = calculate_iou(ex_box, new_box)
                        if iou > iou_limit: # If overlaps more than 45%
                            is_duplicate = True
                            break
                
                # If not a duplicate, add to list
                if not is_duplicate:
                    new_boxes_to_write.append(new_box)

        # 4. APPEND NEW LABELS TO FILE (Append Mode 'a')
        if new_boxes_to_write:
            with open(txt_path, "a") as f: # 'a' append mode: add to end, don't delete
                # If file is not empty and no \n at last line, move to new line first
                if os.path.getsize(txt_path) > 0:
                    f.write("\n") 
                    
                for b in new_boxes_to_write:
                    line = f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n"
                    f.write(line)
                    count_new_labels += 1

    print(f"\nProcess Completed! Total {count_new_labels} new object labels added.")
    print(f"Labels saved to: {labels_dir}")