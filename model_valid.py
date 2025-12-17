from ultralytics import YOLO
import os

def main(model_path, data_yaml, run_name, output_valid):
    
    print("\n-----------------------------------")
    print(f"Starting Validation...")
    print(f"Model Used: {model_path}")
    print(f"Data_yaml path: {data_yaml}")
    print("-----------------------------------")

    # ================================
    # 1. LOAD MODEL
    # ================================
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found -> {model_path}")
        return

    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")

    # ================================
    # 2. DETERMINE FOLDER PATHS
    # ================================
    validation_dir = os.path.join(output_valid, "Model_Validations")
    # Final Directory Path to Save
    save_path = os.path.join(validation_dir, run_name)

    # Create folder if it doesn't exist (Manual check is good even if YOLO does it)
    os.makedirs(save_path, exist_ok=True) 

    # ================================
    # 3. RUN VALIDATION
    # ================================
    try:
        metrics = model.val(
            data=data_yaml,
            split="val",
            workers=0,
            project=validation_dir, # 'Model_Validations' folder
            name=run_name,          # Subfolder where results will be saved
            exist_ok=True
        )

        # Extracting metrics
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        
        # ================================
        # 4. WRITING RESULTS TO TXT FILE (NEWLY ADDED PART)
        # ================================
        
        output_file_path = os.path.join(save_path, "mAP_results.txt")
        
        with open(output_file_path, 'w') as f:
            f.write("========== YOLOv8 VALIDATION METRICS ==========\n")
            f.write(f"Model: {os.path.basename(model_path)}\n")
            f.write(f"Dataset: {os.path.basename(data_yaml)}\n")
            f.write("-------------------------------------------------\n")
            f.write(f"mAP50 (Mean Average Precision @ 0.5 IOU): {map50:.4f}\n")
            f.write(f"mAP50-95 (Mean Average Precision @ 0.5-0.95 IOU): {map50_95:.4f}\n")
            f.write("=================================================\n")

        print(f"Validation results saved:\n-> {save_path}")
        print(f"Metrics saved to TXT file: -> {output_file_path}")
        print("\nmAP50:", map50)
        print("mAP50-95:", map50_95)

    except Exception as e:
        print(f"VALIDATION ERROR: {e}")