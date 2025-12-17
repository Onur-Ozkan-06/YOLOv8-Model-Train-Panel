import os
import shutil
from ultralytics import YOLO

def main(model_path, val_img, output_predict, run_name):
    
    model_predict = os.path.join(output_predict, "Model_Predictions")
    print("\n-----------------------------------")
    print(f"Starting Prediction...")
    print(f"Model Used: {model_path}")
    print(f"Image Path Used: {val_img}")
    print("-----------------------------------")

    # 1. Check if files actually exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found!\nPath: {model_path}")
        return
    
    if not os.path.exists(val_img):
        print(f"ERROR: Validation image path not found!\nPath: {val_img}")
        return
    
    try:
        # Load the model
        model = YOLO(model_path)

        print("Model loaded, starting test (Please wait)...")

        # 2. Start prediction process
        pred_results = model.predict(
            val_img,
            save=True,
            conf=0.4,
            project=model_predict, 
            name=run_name,
            exist_ok=True  
        )   

    except Exception as e:
        # If an error occurs, do not close the program, show the error
        print(f"\n!!! AN UNEXPECTED ERROR OCCURRED !!!")
        print(f"Error details: {e}")
