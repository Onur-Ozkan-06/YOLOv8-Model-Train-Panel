
# 🚀 YOLOv8 Model Training Panel

[![YOLOv8](https://img.shields.io/badge/YOLO-v8-blue)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)

**YOLOv8 Model Training Panel** is a Streamlit-based Graphical User Interface (GUI) that simplifies training Ultralytics YOLOv8 models. No more CLI commands—manage your training, parameters, and datasets through an intuitive web interface.

---

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset Structure](#-dataset-structure)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **Streamlit Interface** | A modern, browser-based dashboard for easy interaction. |
| **One-Click Start** | Quick launch via Windows `.bat` file. |
| **Hyperparameter Tuning** | Adjust Epochs, Batch Size, and Image Size via sliders/inputs. |
| **Automatic GPU Detection** | Automatically uses CUDA if available, otherwise falls back to CPU. |
| **Model Selection** | Choose from different YOLOv8 model sizes (n, s, m, l, x). |

---

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/Onur-Ozkan-06/YOLOv8-Model-Train-Panel.git](https://github.com/Onur-Ozkan-06/YOLOv8-Model-Train-Panel.git)
cd YOLOv8-Model-Train-Panel

```

### 2. Set Up Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## 🚀 Usage

Since this is a Streamlit application, you **cannot** run it using `python App.py`. Use one of the following methods:

### Option A: One-Click Start (Windows)

1. Navigate to the project folder.
2. Double-click the `.bat` file (e.g., `run_panel.bat`) to launch the app automatically.

### Option B: Manual Command Line

1. Open your terminal in the project directory.
2. Ensure your virtual environment is active.
3. Run the following command:
```bash
streamlit run App.py

```



---

## 📂 Dataset Structure

Your dataset must be organized in the YOLO format:

```text
DatasetFolder/
├── data.yaml          # Configuration file
├── train/
│   ├── images/        # Training images
│   └── labels/        # Training labels (.txt)
└── valid/
    ├── images/        # Validation images
    └── labels/        # Validation labels (.txt)

```

**Inside your `data.yaml`:**

```yaml
path: ../DatasetFolder
train: train/images
val: valid/images
names:
  0: class_name_1
  1: class_name_2

```


### 📌 How to Create a Desktop Shortcut

> **⚠️ IMPORTANT:** Do not move the `.bat` file out of the project folder!
> If you move the file itself to the Desktop, it will lose the connection to `app.py`, and the application will fail to start.

**To run the app from your Desktop correctly:**

1.  **Right-click** on the `.bat` file inside the project folder.
2.  Select **Send to** > **Desktop (create shortcut)**.
3.  You can now safely rename or move this **shortcut** anywhere you like.

```

### Recommendation for your `.bat` file

To make your batch file even more robust (just in case someone ignores the warning), I strongly recommend updating the content of your `.bat` file to this. The command `cd /d "%~dp0"` forces the script to look inside its own folder, no matter where it is called from:

```batch
@echo off
:: Fix character encoding
chcp 65001 > nul

:: Set the working directory to the folder where this file is located
cd /d "%~dp0"

:: Activate venv if it exists (optional)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: Run the app
python -m streamlit run app.py
pause

```




## 🔧 Troubleshooting

* **App won't start with `python App.py`:** This is expected. Always use `streamlit run App.py`.
* **'streamlit' is not recognized:** Ensure you have installed the requirements using `pip install -r requirements.txt` and your virtual environment is active.
* **GPU not working:** Verify your PyTorch installation with CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"

```



---

## 🤝 Contributing

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.
