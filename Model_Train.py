from ultralytics import YOLO
import torch
import os
from pathlib import Path

# PARAMETERS ADDED: model_name, epochs, batch_size
def main(data_yaml, run_name, output_model, model_name="yolov8s.pt", epochs=100, batch_size=16, **kwargs):
    # ==========================================
    # 1. SETTINGS AND PATH CALCULATION
    # ==========================================
    
    # output_model path → let's find the project folder
    target_file = Path(output_model)
    project_dir = target_file.parent.parent.parent  # weights → run_name → runs/detect

    # GPU Check
    # 
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = 0
        device_info = f"GPU ACTIVE: {device_name}"
    else:
        device = 'cpu'
        device_info = "GPU NOT FOUND, USING CPU"

    # ==========================================
    # 2. INFORMATION
    # ==========================================
    print("\n" + "="*50)
    print(f"   PPE TRAINING MODULE STARTING")
    print("="*50)
    print(f"Model Name             : {run_name}")
    print(f"Base Model             : {model_name}")  # Added
    print(f"Epoch / Batch          : {epochs} / {batch_size}") # Added
    print(f"Hardware               : {device_info}")
    print(f"Dataset YAML           : {data_yaml}")
    print(f"Save Location (Project): {project_dir}")
    print("-" * 50)
    print("Training starting...")
    print("-" * 50)

    # ==========================================
    # 3. MODEL LOADING AND TRAINING
    # ==========================================
    try:
        # LOADING SELECTED MODEL
        # 
        model = YOLO(model_name)

        results = model.train(
            data=data_yaml,
            epochs=epochs,          # From user
            patience=25,
            imgsz=640,
            batch=batch_size,       # From user
            workers=0,
            device=device,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            **kwargs
        )

        print("\n" + "="*50)
        print("   TRAINING COMPLETED")
        print("="*50)

    except Exception as e:
        print(f"\n!!! ERROR OCCURRED DURING TRAINING !!!\n{e}")
        return

    # ==========================================
    # 4. POST-TRAINING → VALIDATION
    # ==========================================
    if os.path.exists(output_model):
        print(f"Success! Model saved:\n-> {output_model}")

        print("\n--- Validation (model.val) Starting ---")

        try:
            trained_model = YOLO(output_model)

            # VALIDATION save folder
            model_validations = os.path.join(project_dir, "Model_Validations")

            # Run validation

            metrics = trained_model.val(
                data=data_yaml,
                split="val",
                workers=0,
                project=model_validations,   # → runs/detect/Model_Validations
                name=run_name,               # → subfolder: run_name
                exist_ok=True
            )

            print(f"Validation results saved:\n-> {os.path.join(model_validations, run_name)}")
            print("\nmAP50:", metrics.box.map50)
            print("mAP50-95:", metrics.box.map)

        except Exception as e:
            print(f"Error during validation: {e}")

    else:
        print(f"\nWARNING: Training finished but {output_model} file not found.")
        print("Please check the 'runs/detect' folder manually.")