import json
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# --- Turkish Character Correction (Windows) ---
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ===== Config Operations =====
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {"dataset_path": "", "data_yaml": "", "val_images": "", "model_path": "runs/detect"}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    folder = os.path.dirname(CONFIG_PATH)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

config = load_config()

# ===== Path Definition =====
def define_paths():
    print("\n--- PATH DEFINITION ---")
    config["dataset_path"] = input(f"Dataset Path [{config.get('dataset_path', '')}]: ") or config.get("dataset_path", "")
    config["data_yaml"] = input(f"Data Yaml Path [{config.get('data_yaml', '')}]: ") or config.get("data_yaml", "")
    config["val_images"] = input(f"Val Images Path [{config.get('val_images', '')}]: ") or config.get("val_images", "")
    config["model_path"] = input(f"Model Base Path (runs/detect) [{config.get('model_path', '')}]: ") or config.get("model_path", "")
    save_config(config)
    print("✅ Settings saved.")

# -------- Helper Functions ----------
def best_pt(run_name):
    # Model weight file (Source) - DO NOT DELETE
    base = config["model_path"]
    return os.path.join(base, run_name, "weights", "best.pt")

def force_clean_dir(directory_path):
    """Deletes the given folder without confirmation."""
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"🗑️  Old results cleaned: {directory_path}")
        except Exception as e:
            print(f"⚠️  Deletion error (File might be open): {e}")

# ====================== MENU FUNCTIONS =========================

def list_models():
    base = config["model_path"]
    if not os.path.exists(base):
        print(f"Model folder not found: {base}")
        return []
    models = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    # Exclude Model_Validations and Model_Predictions folders from list (to avoid confusion)
    exclude_list = ["Model_Validations", "Model_Predictions", "val_exp", "predict_exp"]
    models = [m for m in models if m not in exclude_list]
    
    if not models:
        print("No models found.")
        return []
    for i, m in enumerate(models, 1):
        print(f"{i}) {m}")
    return models

def select_model():
    models = list_models()
    if not models: return None
    sec = input("\nSelect model (name or number): ")
    if sec.isdigit():
        idx = int(sec) - 1
        if 0 <= idx < len(models): return models[idx]
    if sec in models: return sec
    print("Invalid selection.")
    return None

# --- 1. TRAINING ---
def ppe_training():
    # Keeping imports same to avoid breaking code
    from PPE_Egitim import main 
    
    # 1. Determine Model Name
    while True:
        run_name = input("New Model Folder Name for Training (Ex: Test_V1): ")
        if not run_name: return
        
        target_dir = os.path.join(config["model_path"], run_name)
        if os.path.exists(target_dir):
            print(f"❌ ERROR: A model named '{run_name}' already exists! Enter a different name.")
        else:
            break

    # 2. Select Model Structure (n, s, m, l, x)
    print("\n--- Select Model Structure ---")
    print("1) yolov8n.pt (Nano   - Fastest, Low Accuracy, for Mobile/RPi)")
    print("2) yolov8s.pt (Small  - Balanced, Ideal for RPi4/Jetson)")
    print("3) yolov8m.pt (Medium - Slow, High Accuracy, for PC/Cloud)")
    print("4) yolov8l.pt (Large  - Very Slow, Highest Accuracy)")
    
    model_choice = input("Select model (1-4) [Default: 2]: ")
    
    if model_choice == "1":
        selected_model = "yolov8n.pt"
        rec_batch = "16 or 24"
        default_batch = 16
    elif model_choice == "3":
        selected_model = "yolov8m.pt"
        rec_batch = "4 or 8"
        default_batch = 4
    elif model_choice == "4":
        selected_model = "yolov8l.pt"
        rec_batch = "4 or 8"
        default_batch = 4
    else:
        selected_model = "yolov8s.pt" # Default
        rec_batch = "16 or 24"
        default_batch = 16

    print(f"✅ Selected Model: {selected_model}")

    # 3. Epoch Count
    try:
        ep_input = input(f"Epoch Count [Default: 100]: ")
        epochs = int(ep_input) if ep_input.strip() else 100
    except ValueError:
        print("Invalid input, set to default 100 epochs.")
        epochs = 100

    # 4. Batch Size (With recommendation)
    try:
        print(f"\nℹ️  Recommended Batch Size for {selected_model}: {rec_batch}")
        bs_input = input(f"Enter Batch Size [Default: {default_batch}]: ")
        batch_size = int(bs_input) if bs_input.strip() else default_batch
    except ValueError:
        print(f"Invalid input, set to default batch {default_batch}.")
        batch_size = default_batch

    # Start Training
    model_path = best_pt(run_name)
    main(
        data_yaml=config["data_yaml"], 
        run_name=run_name, 
        output_model=model_path,
        model_name=selected_model, 
        epochs=epochs,            
        batch_size=batch_size     
    )

# --- 2. FINETUNE ---
def ppe_finetune():
    from PPE_Finetune import main
    print("\n--- Base Model ---")
    base_model = select_model()
    if not base_model: return
    
    run_name = f"{base_model}_Finetune"
    print(f"\n✅ New Model Name Determined Automatically: {run_name}")

    # Finetune creates a new model, so we check the main folder
    target_dir = os.path.join(config["model_path"], run_name)
    force_clean_dir(target_dir) # Delete if exists

    main(
        data_yaml=config["data_yaml"],
        run_name=run_name,
        model_path=best_pt(base_model),
        output_model=best_pt(run_name)
    )

# --- 3. VALIDATION ---
def model_valid():
    from model_valid import main
    run_name = select_model()
    if not run_name: return
    
    # TARGET: runs/detect/Model_Validations/Test4
    results_dir = os.path.join(config["model_path"], "Model_Validations", run_name)
    
    # Clean only the result folder
    force_clean_dir(results_dir)

    main(
        model_path=best_pt(run_name),
        data_yaml=config["data_yaml"],
        run_name=run_name,
        output_valid=config["model_path"] # Script likely creates Model_Validations inside this
    )

# --- 4. PREDICTION ---
def model_predict():
    from model_predict import main
    run_name = select_model()
    if not run_name: return
    
    # TARGET: runs/detect/Model_Predictions/Test4
    results_dir = os.path.join(config["model_path"], "Model_Predictions", run_name)
    
    # Clean only the result folder
    force_clean_dir(results_dir)

    main(
        model_path=best_pt(run_name),
        val_img=config["val_images"],
        output_predict=config["model_path"],
        run_name=run_name
    )

# ... Other Functions ...
def class_weight_detection():
    from Class_Weight_Tespit import main
    main(config["dataset_path"])

def solo_ppe_detection():
    from Solo_person_ppe_tespit import main
    run_name = select_model()
    if run_name: main(dataset_path=config["dataset_path"], model_path=best_pt(run_name))

def ppe_detection():
    from PPE_Tespit import main
    run_name = select_model()
    if run_name:
        img_path = input("Image Path: ").strip('"').strip("'")
        if os.path.exists(img_path): main(model_path=best_pt(run_name), img_path=img_path)
        else: print("Image not found.")

def pseudo_label():
    from Pseudo_labeling import main
    run_name = select_model()
    if not run_name : return
    model_path = best_pt(run_name)
    main(
        model_path = model_path,
        dataset_path = config["dataset_path"],
        val_images = config["val_images"]
    )

    

# ====================== MAIN MENU =========================
def menu():
    while True:
        print("""
==================================
      PPE DETECTION SYSTEM (CLI)
==================================
1 - Define Path
2 - Class Weight Detection
3 - PPE Training
4 - PPE Finetune 
5 - Model Validation 
6 - Model Prediction
7 - Solo Person PPE
8 - Single Image Detection
9 - Pseudo Label
0 - Exit
""")
        choice = input("Selection: ")
        try:
            if choice == "1": define_paths()
            elif choice == "2": class_weight_detection()
            elif choice == "3": ppe_training()
            elif choice == "4": ppe_finetune()
            elif choice == "5": model_valid()
            elif choice == "6": model_predict()
            elif choice == "7": solo_ppe_detection()
            elif choice == "8": ppe_detection()
            elif choice == "9": pseudo_label()
            elif choice == "0": break
            else: print("Invalid selection.")
        except Exception as e:
            print(f"\n⚠️ ERROR: {e}")

if __name__ == "__main__":
    menu()