from ultralytics import YOLO
import torch
import os
from pathlib import Path

def main(data_yaml, run_name, model_path, output_model, epochs, batch_size, **kwargs):

    # ==========================================
    # 1. PATH & HARDWARE CHECKS
    # ==========================================

    target_file = Path(output_model)
    project_dir = target_file.parent.parent.parent  # weights -> RunName -> runs/detect

    # Is there a base model for fine-tuning?
    # [Image of transfer learning diagram in neural networks]

    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Required base model for Fine-tuning not found!")
        print(f"Searched Path : {model_path}")
        return

    # GPU check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = 0
        device_info = f"GPU ACTIVE: {device_name}"
    else:
        device = 'cpu'
        device_info = "NO GPU, Using CPU"

    # ==========================================
    # 2. INFORMATION
    # ==========================================
    print("\n" + "="*60)
    print(f"      PPE FINETUNE MODULE STARTING")
    print("="*60)
    print(f"Base Model (Old):  {model_path}")
    print(f"New Model (Run):   {run_name}")
    print(f"Hardware        :  {device_info}")
    print(f"Save Location   :  {project_dir}")
    print("-" * 60)
    print("Fine-tune process starting...")
    print("-" * 60)

    # ==========================================
    # 3. FINE-TUNE TRAINING PHASE
    # ==========================================

    try:
        model = YOLO(model_path)

        IMG_SIZE = 640

        # 
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=IMG_SIZE,
            batch=batch_size,
            device=device,
            workers=0,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            optimizer='AdamW',
            cls=1.5,
            box=7.5,
            dfl=1.0,
            **kwargs
        )

        print("\n" + "="*60)
        print("        🎉 FINETUNE COMPLETED SUCCESSFULLY")
        print("="*60)

    except Exception as e:
        print(f"\n❌ FINETUNE ERROR:\n{e}")
        return

    # ==========================================
    # 4. VALIDATION 
    # ==========================================

    if os.path.exists(output_model):
        print(f"✔ New model saved : {output_model}")

        print("\n--- VALIDATION STARTING ---")

        try:
            finetuned_model = YOLO(output_model)

            # Folder for validation results
            validation_project = os.path.join(project_dir, "Model_Validations")

            # [Image of precision recall curve]

            metrics = finetuned_model.val(
                data=data_yaml,
                split='val',
                imgsz=640,
                workers=0,
                device=device,
                project=validation_project,
                name=run_name,
                exist_ok=True
            )

            print(f"Validation results saved:\n-> {os.path.join(validation_project, run_name)}")
            print("\nmAP50:", metrics.box.map50)
            print("mAP50-95:", metrics.box.map)

        except Exception as e:
            print(f"\n❌ VALIDATION ERROR:\n{e}")

    else:
        print(f"\n⚠️ WARNING: {output_model} not found! Training finished but model might not have been saved.")