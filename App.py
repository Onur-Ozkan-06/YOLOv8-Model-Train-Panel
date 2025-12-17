import streamlit as st
import json
import os
import shutil
import sys
import re
import time

# ================= SETTINGS & SETUP =================

# --- 1. Logger Class to Redirect Terminal Output to Screen ---
class StreamlitLogger:
    def __init__(self, log_placeholder, status_placeholder):
        self.log_placeholder = log_placeholder       # Past logs (Stays at top - Scrollable)
        self.status_placeholder = status_placeholder # Animated bar (Changes at bottom - Progress Bar)
        self.terminal = sys.stdout
        self.log_history = ""
        self.current_line = "" 
        self.ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')

    def write(self, message):
        self.terminal.write(message) # Write to background (real terminal) as well
        
        # Clean ANSI color codes
        clean = self.ansi_escape.sub('', message)
        
        # If message contains a newline (\n) -> Save to history
        if '\n' in clean:
            parts = clean.split('\n')
            
            # Embed all parts except the last one into history
            for p in parts[:-1]:
                if (self.current_line + p).strip(): # Filter empty lines
                    self.log_history += self.current_line + p + "\n"
                self.current_line = "" # Reset line
            
            # Make the remaining last part the current line
            self.current_line += parts[-1]
            
            # Update history (Show last 5000 chars to avoid memory bloat)
            self.log_placeholder.code(self.log_history[-5000:], language="text")
            
        # If message contains carriage return (\r) (YOLO Progress Bar) -> Update only the bottom bar
        elif '\r' in clean:
            # Clean the \r part and update the current line
            self.current_line = clean.replace('\r', '')
        
        else:
            self.current_line += clean

        # Update Animated Bar (Bottom box)
        self.status_placeholder.code(self.current_line, language="text")

    def flush(self):
        self.terminal.flush()

# --- 2. Basic Path Settings ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# --- 3. Turkish Character Correction (Windows) ---
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ================= HELPER FUNCTIONS =================

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

def best_pt(run_name, base_path):
    return os.path.join(base_path, run_name, "weights", "best.pt")

def force_clean_dir(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            st.warning(f"Old folder cleaned: {directory_path}")
            time.sleep(0.5) 
        except Exception as e:
            st.error(f"Deletion error: {e}")

def get_model_list(model_base_path):
    if not os.path.exists(model_base_path):
        return []
    models = [d for d in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path, d))]
    # Do not show result folders in the model list
    exclude_list = ["Model_Validations", "Model_Predictions", "val_exp", "predict_exp"]
    models = [m for m in models if m not in exclude_list]
    return models

# --- ADVANCED PARAMETERS INTERFACE ---
def get_advanced_params():
    """Gets Augmentation and Hyperparameters from the user."""
    params = {}
    
    with st.expander("Advanced Settings (Augmentation & Hyperparameters)", expanded=False):
        
        tab_aug, tab_hyp = st.tabs(["Augmentation", "Hyperparameters"])
        
        # --- 1. Augmentation ---
        with tab_aug:
            st.info("Artificially augments the dataset to increase model success.")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Geometric Changes**")
                params["degrees"] = st.slider("Rotation (Degrees)", 0.0, 180.0, 0.0, 1.0, help="If too high, objects appear at unusual angles.")
                params["translate"] = st.slider("Translate", 0.0, 1.0, 0.1, 0.05, help="Shifts images and objects by a certain fraction.")
                params["scale"] = st.slider("Scale", 0.0, 1.0, 0.5, 0.1, help="Randomly changes object size.")
                params["shear"] = st.slider("Shear", 0.0, 45.0, 0.0, 1.0, help="Skews images in one direction (If too high, object shapes get distorted).")
                params["fliplr"] = st.slider("Horizontal Flip (Flip LR)", 0.0, 1.0, 0.5, 0.1, help="Flips the image horizontally.")

            with col2:
                st.markdown("**Color and Mixing**")
                params["mosaic"] = st.slider("Mosaic", 0.0, 1.0, 0.2, 0.1, help="Combines 4 images (Too high disrupts dataset structure).")
                params["mixup"] = st.slider("Mixup", 0.0, 1.0, 0.0, 0.1, help="Linearly combines two images and labels (Too high confuses image and label).")
                st.markdown("---")
                params["hsv_h"] = st.slider("HSV Hue", 0.0, 1.0, 0.015, 0.001, help="Too high distorts colors significantly.")
                params["hsv_s"] = st.slider("HSV Saturation", 0.0, 1.0, 0.2, 0.05, help="Too high makes images very pale or overly vivid.") 
                params["hsv_v"] = st.slider("HSV Value (Brightness)", 0.0, 1.0, 0.2, 0.05, help="Too high makes images very bright or dark.")

        # --- 2. Hyperparameters ---
        with tab_hyp:
            st.info("Determines the model's learning strategy.")
            c1, c2 = st.columns(2)
            with c1:
                params["lr0"] = st.number_input("Initial Learning Rate (lr0)", value=0.01, format="%.4f", step=0.001, help="Model learning speed (Too high causes unstable training).")
                params["lrf"] = st.number_input("Final Learning Rate (lrf)", value=0.01, format="%.4f", step=0.001, help="Final Learning Rate Multiplier (Too high slows down learning rate decay, risking overfitting).")
                params["momentum"] = st.number_input("Momentum", value=0.937, format="%.3f", step=0.001)
            
            with c2:
                params["weight_decay"] = st.number_input("Weight Decay", value=0.0005, format="%.5f", step=0.0001, help="Penalty term controlling model weight size (Too high reduces learning ability).")
                params["warmup_epochs"] = st.number_input("Warmup Epochs", value=3.0, step=1.0)
                params["cache"] = st.checkbox("Cache Images (RAM Usage)", value=False, help="If enabled, training speeds up but RAM fills up.")

    return params

# ================= UI (STREAMLIT) START =================

st.set_page_config(page_title="Object Detection & YOLO Training Panel", layout="wide")
st.title("Object Detection & YOLO Training Panel")

# Load Config
if 'config' not in st.session_state:
    st.session_state.config = load_config()
config = st.session_state.config

# --- SIDEBAR ---
st.sidebar.header("Settings")

with st.sidebar.expander("File Paths", expanded=False):
    with st.form("path_form"):
        new_dataset = st.text_input("Dataset Path", config.get("dataset_path", ""))
        new_yaml = st.text_input("Data YAML Path", config.get("data_yaml", ""))
        new_val = st.text_input("Val Images Path", config.get("val_images", ""))
        new_model_dir = st.text_input("Model Base Path", config.get("model_path", "runs/detect"))
        
        if st.form_submit_button("Save Settings"):
            config["dataset_path"] = new_dataset
            config["data_yaml"] = new_yaml
            config["val_images"] = new_val
            config["model_path"] = new_model_dir
            save_config(config)
            st.session_state.config = config
            st.success("Settings saved!")

page = st.sidebar.radio("Select Operation", 
    ["Home", "Class Weight", "Training", "Finetune", "Validation", "Prediction", "Object Detection", "Pseudo Labeling"])

# ================= PAGE LOGIC =================

# ================= HOME (GUIDE) =================
if page == "Home":
    st.markdown("<h1 style='text-align: center;'>Dataset and Setup Guide</h1>", unsafe_allow_html=True)
    
    st.info("Welcome! For the system to work correctly, your dataset must be in the following format.")

    # --- 1. Dataset Folder Structure ---
    st.subheader("1. How Should the Folder Structure Be?")
    st.write("For YOLOv8 training, your folders should look exactly like this:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Folder Tree**")
        st.code("""
MY_DATASET/
│
├── data.yaml  <-- (Most Important File)
│
├── train/
│   ├── images/  (Training images .jpg .png)
│   └── labels/  (Labels in YOLO format .txt)
│
└── valid/
    ├── images/  (Validation images)
    └── labels/  (Validation labels)
""", language="text")
        st.caption("Tip: 'images' and 'labels' folder names must be lowercase and in English.")

    with col2:
        st.markdown("**data.yaml Content**")
        st.code("""
# data.yaml example
                
path: C:/Users/User/Dataset
train: train/images
val: valid/images

nc: 3            
names: ['Helmet', 'Vest', 'No-Vest']  # Class Names
""", language="yaml")
        st.caption("Tip: Using Full Paths in the YAML file reduces errors.")

    st.markdown("---")

    # --- 2. Path Entry Rules ---
    st.subheader("2. How to Enter File Paths?")
    
    st.warning("If using Windows, pay attention to quotation marks when copying file paths!")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.error("WRONG USAGE")
        st.code('"C:\\Users\\Onur\\Desktop\\Dataset"', language="bash")
        st.code('"C:\\Users\\Onur\\Desktop\\Dataset\\data.yaml"', language="bash")
        st.code('"C:\\Users\\Onur\\Desktop\\Dataset\\valid\\images"', language="bash")
        st.write("* Quotation marks ( \" ) should not be present.")
        st.write("* There should be no trailing spaces.")

    with c2:
        st.success("CORRECT USAGE")
        st.code('C:\\Users\\Onur\\Desktop\\Dataset', language="bash")
        st.code('C:\\Users\\Onur\\Desktop\\Dataset\\data.yaml', language="bash")
        st.code('C:\\Users\\Onur\\Desktop\\Dataset\\valid\\images', language="bash")
        st.write("* Quotation marks cleaned.")
        st.write("* English characters are safest, though our system handles corrections.")

    st.markdown("---")

    # --- 3. Quick Start ---
    st.subheader("Quick Start Steps")
    st.markdown("""
    1. Go to the **Settings** section in the left menu.
    2. Enter the paths for your Dataset and `data.yaml` file.
    3. Enter the path for Dataset/valid/images within your dataset.
    4. Enter the location where trained and used models will be saved in the Model Base Path section.
    5. Click the **"Save Settings"** button.
    6. Switch to the **"Training"** tab and start training your model!
    """)

# --- CLASS WEIGHT ---
elif page == "Class Weight":
    st.header("Class Weight Calculation")
    st.write("In this menu, you can calculate the Class weights of your Dataset.")
    
    st.write("### Logs")
    log_area = st.empty()
    status_area = st.empty()
    
    if st.button("Start Calculation"):
        sys.stdout = StreamlitLogger(log_area, status_area)
        try:
            # NOTE: Keeping the original file import name to avoid breakage
            from Class_Weight_Detection import main as cw_main
            cw_main(config["dataset_path"])
            st.success("Process completed!")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            sys.stdout = sys.__stdout__

# --- TRAINING ---
elif page == "Training":
    st.header("Start New Training")

    # 1. Model Name
    run_name = st.text_input("Model Name (Run Name)", "Experiment_1", help="Will be saved under Model Base Path with this name.")
    
    # 2. Basic Parameters
    st.subheader("Basic Parameters")
    
    model_options = {
        "yolov8n.pt": "Nano (Fast)",
        "yolov8s.pt": "Small (Balanced)",
        "yolov8m.pt": "Medium (Strong)",
        "yolov8l.pt": "Large (Heavy)"
    }

    selected_model_key = st.selectbox(
        "Select Model Structure",
        options=list(model_options.keys()),
        format_func=lambda x: f"{x} - {model_options[x]}",
        index=1
    )

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epoch Count", min_value=1, max_value=3000, value=100, step=10)
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=16, step=1)

    # 3. Advanced Parameters (Augmentation)
    advanced_params = get_advanced_params()

    # Folder Check
    target_dir = os.path.join(config["model_path"], run_name)
    folder_exists = os.path.exists(target_dir)
    
    if folder_exists:
        st.error(f"ERROR: A model named '{run_name}' already exists! Please change the name.")
    
    st.write("### Training Logs")
    log_area = st.empty()
    status_area = st.empty()
    
    if st.button("Start Training", disabled=folder_exists):
        # NOTE: Keeping original import
        from Model_Train import main as train_main
        model_out = best_pt(run_name, config["model_path"])
        
        sys.stdout = StreamlitLogger(log_area, status_area)
        try:
            with st.spinner(f"Starting Training... Model: {selected_model_key}"):
                # Sending parameters to backend (**advanced_params)
                train_main(
                    data_yaml=config["data_yaml"], 
                    run_name=run_name, 
                    output_model=model_out,
                    model_name=selected_model_key,
                    epochs=epochs,
                    batch_size=batch_size,
                    **advanced_params 
                )
            
            st.success(f"Training Completed Successfully! ({run_name})")
            st.balloons()
        except Exception as e:
            st.error(f"Error occurred during training: {e}")
        finally:
            sys.stdout = sys.__stdout__

# --- FINETUNE (Auto Name + Delete + Augmentation) ---
elif page == "Finetune":
    st.header("Fine-Tune")
    models = get_model_list(config["model_path"])
    base_model = st.selectbox("Base Model", models)
    
    st.subheader("Parameters")
    col1,col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epoch Count", min_value=1, max_value=1000, value=100, step=10)
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=4, step=1)

    # Advanced Parameters
    advanced_params = get_advanced_params()

    if base_model:
        # Auto Naming
        run_name = f"{base_model}_Finetune"
        st.info(f"New Model Name: **{run_name}**")
        st.warning(f"If '{run_name}' folder exists, CONTENTS WILL BE DELETED.")
        
        st.write("### Logs")
        log_area = st.empty()
        status_area = st.empty()
        
        if st.button("Start Finetune"):
            target_dir = os.path.join(config["model_path"], run_name)
            force_clean_dir(target_dir) # Auto Clean
            
            # NOTE: Keeping original import
            from Model_Finetune import main as ft_main
            base_pt = best_pt(base_model, config["model_path"])
            out_pt = best_pt(run_name, config["model_path"])
            
            sys.stdout = StreamlitLogger(log_area, status_area)
            try:
                with st.spinner('Finetuning in progress...'):
                    ft_main(
                        data_yaml=config["data_yaml"], 
                        run_name=run_name, 
                        model_path=base_pt, 
                        output_model=out_pt, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        **advanced_params 
                    )
                st.success("Finetune Completed!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                sys.stdout = sys.__stdout__

# --- VALIDATION ---
elif page == "Validation":
    st.header("Model Validation")
    models = get_model_list(config["model_path"])
    run_name = st.selectbox("Select Model", models)
    
    st.info("Caution: When the process starts, old validation results created with this model will be deleted.")
    
    st.write("### Logs")
    log_area = st.empty()
    status_area = st.empty()
    
    if st.button("Run Validation"):
        # Folder: runs/detect/Model_Validations/ModelName
        results_dir = os.path.join(config["model_path"], "Model_Validations", run_name)
        force_clean_dir(results_dir)
        
        # NOTE: Keeping original import
        from model_valid import main as valid_main
        model_pt = best_pt(run_name, config["model_path"])
        
        sys.stdout = StreamlitLogger(log_area, status_area)
        try:
            with st.spinner('Validating...'):
                valid_main(model_path=model_pt, data_yaml=config["data_yaml"], run_name=run_name, output_valid=config["model_path"])
            st.success("Validation completed.")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            sys.stdout = sys.__stdout__

# --- PREDICTION ---
elif page == "Prediction":
    st.header("Model Prediction")
    models = get_model_list(config["model_path"])
    run_name = st.selectbox("Select Model", models)
    
    st.info("Caution: When the process starts, old prediction results created with this model will be deleted.")
    
    st.write("### Logs")
    log_area = st.empty()
    status_area = st.empty()
    
    if st.button("Run Prediction"):
        # Folder: runs/detect/Model_Predictions/ModelName
        results_dir = os.path.join(config["model_path"], "Model_Predictions", run_name)
        force_clean_dir(results_dir)
        
        # NOTE: Keeping original import
        from model_predict import main as pred_main
        model_pt = best_pt(run_name, config["model_path"])
        
        sys.stdout = StreamlitLogger(log_area, status_area)
        try:
            with st.spinner('Predicting...'):
                pred_main(model_path=model_pt, val_img=config["val_images"], output_predict=config["model_path"], run_name=run_name)
            st.success("Prediction completed.")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            sys.stdout = sys.__stdout__

# --- Object Detection (Single Image) ---
elif page == "Object Detection":
    st.header("Single Image Test")
    
    # Check if models exist
    models = get_model_list(config["model_path"])
    
    if not models:
        st.warning("⚠️ No models found! Please train a model first or check your 'Model Base Path'.")
        selected_model = None
    else:
        selected_model = st.selectbox("Select Model", models)
    
    tab1, tab2 = st.tabs(["Upload File", "Enter Path"])
    temp_path = None
    
    with tab1:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            temp_path = os.path.join(BASE_DIR, "temp_image.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    
    with tab2:
        path_input = st.text_input("Image Path", "")
        if path_input:
            clean_path = path_input.strip('"').strip("'")
            if os.path.exists(clean_path):
                temp_path = clean_path
            else:
                st.error("File not found!")

    st.write("### Logs")
    log_area = st.empty()
    status_area = st.empty()
    
    # Added check: 'and selected_model'
    if temp_path and st.button("Analyze"):
        if selected_model:
            from Model_Predict_img import main as ppe_main
            model_pt = best_pt(selected_model, config["model_path"])
            
            sys.stdout = StreamlitLogger(log_area, status_area)
            try:
                st.write("Analyzing...")
                ppe_main(model_path=model_pt, img_path=temp_path)
                st.success("Process finished.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                sys.stdout = sys.__stdout__
        else:
            st.error("Please select a valid model first!")


# --- PSEUDO LABELING ---
elif page == "Pseudo Labeling":
    st.header("AI Assisted Auto Labeling (Pseudo-Labeling)")
    
    st.markdown("""
    **What Does This Module Do?**
    1. Uses an existing trained model to automatically label **new unlabeled photos**.
    2. If the photo already has manually drawn labels, **it does not delete them**.
    3. **Smart Overlap Prevention:** If the object found by the model overlaps with your label (Duplicate), it cancels the model's prediction. This prevents double boxes.
    """)
    
    st.info("Caution: This process writes to the `dataset/train/labels` folder. It is recommended to backup your original data.")

    # 1. Model Selection
    models = get_model_list(config["model_path"])
    
    # --- CRITICAL FIX START ---
    if not models:
        st.warning("⚠️ No models found in the directory! You need a trained model to perform Pseudo-Labeling.")
        st.stop() # Stops execution here so it doesn't crash below
    # --- CRITICAL FIX END ---

    selected_model = st.selectbox("Expert Model to Use", models)
    
    # Now it is safe to call best_pt because we know selected_model is not None
    model_pt = best_pt(selected_model, config["model_path"])
    
    st.markdown("---")
    
    # 2. Dataset Settings
    st.subheader("Target Folder")
    target_images_path = st.text_input(
        "Folder of Images to Label", 
        value=os.path.join(config.get("dataset_path", ""), "train", "images"),
        help="Ex: C:/Dataset/Project/train/images"
    )

    st.markdown("---")
    
    # ... (Kodun geri kalanı aynı) ...
    # 3. Class Information ve devamı...
    # ...
    # ...