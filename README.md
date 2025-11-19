

# **BMOD File Processing Studio**

A lightweight Gradio interface for uploading, extracting, and preparing ZIP-based datasets inside a workspace directory.
Ideal as a pre-processing tool for OCR pipelines, log analysis, or any batch-processing workflow.

## Live
https://huggingface.co/spaces/NauRaa/OCR

---

## 🚀 **Overview**

This project provides a very simple UI and reliable interface that allows the user to:

* Upload a **ZIP file**
* Automatically extract it into a workspace
* Run a **processing step** (placeholder logic for now)
* Prepare the extracted files for later use inside a notebook or server-side pipeline

🟢 **No machine learning model is used here yet.**
This tool is purely a **file preparation / extraction pipeline** designed to work even on **CPU-only environments** such as free Hugging Face Spaces.

It works smoothly on:

* Google Colab using TrOCR
* Hugging Face Spaces with easyOCR
* Local machines

---

## 🧩 **Why This Tool Exists?**

During development, we found that:

* Hugging Face free tier **does not allow GPU**
* Long-running scripts or heavy models can **timeout**
* Uploading many files individually is slow
* Logs and OCR datasets usually come as nested folder structures


The idea is:
👉 **Prepare your dataset cleanly first**, then run the model separately in Colab or locally.

---

## 📂 **Project Structure**

When the interface runs, it automatically creates:

```
workspace/
 ├── uploaded/      # Extracted ZIP contents
 └── processed/     # Output folder after processing step
```

All uploads and outputs stay isolated inside `workspace/`.

---

## ⚙️ **Features**

| Feature              | Description                                             |
| -------------------- | ------------------------------------------------------- |
| ZIP Upload           | Upload any .zip file via interface                      |
| Extraction           | Unzips directly into workspace/uploaded                 |
| Cleanup              | Automatically deletes old uploads before new extraction |
| Processing           | Placeholder example: copies files to /processed         |
| CPU Friendly         | Works with zero GPU requirements                        |
| Ready for Deployment | Runs smoothly on Hugging Face Spaces                    |


## 🖼️ **Gradio UI**

The interface includes:

* File upload box
* "Extract ZIP" button
* "Run Processing" button
* Status text outputs

User-friendly and works directly inside the browser.

---

## 🧾 **Code**

```python
import gradio as gr
import os
import zipfile
import shutil

# ------------------------------
# 1) Folder Structure Management
# ------------------------------
WORK_DIR = "workspace"
UPLOAD_DIR = os.path.join(WORK_DIR, "uploaded")
PROCESSED_DIR = os.path.join(WORK_DIR, "processed")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ------------------------------
# 2) Extract ZIP
# ------------------------------
def extract_zip(zip_file):
    if zip_file is None:
        return "❌ Please upload a ZIP file first."

    # Clear old uploads
    shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save the uploaded file
    zip_path = os.path.join(UPLOAD_DIR, "data.zip")
    zip_file.save(zip_path)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(UPLOAD_DIR)

    return "✅ ZIP extracted successfully. Files are ready."

# ------------------------------
# 3) Process Files (Dummy Example)
# ------------------------------
def process_files():
    if not os.listdir(UPLOAD_DIR):
        return "❌ No files found. Did you extract a ZIP?"

    # Clear processed folder
    shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Demo processing: simply copy the extracted files
    shutil.copytree(UPLOAD_DIR, PROCESSED_DIR, dirs_exist_ok=True)

    return f"✅ Processing completed.\n📁 Files processed: {len(os.listdir(PROCESSED_DIR))}"

# ------------------------------
# 4) UI
# ------------------------------
with gr.Blocks(title=\"BMOD File Processing Studio\") as demo:
    gr.Markdown(\"\"\"
    # 📄 **BMOD File Processing Studio**
    Upload a ZIP file, extract it, and run basic processing steps.
    \"\"\")

    with gr.Row():
        zip_input = gr.File(label=\"Upload ZIP\", file_types=[\".zip\"])
        extract_btn = gr.Button(\"Extract ZIP\")

    extract_output = gr.Textbox(label=\"Status\")
    extract_btn.click(extract_zip, inputs=zip_input, outputs=extract_output)

    process_btn = gr.Button(\"Run Processing\")
    process_output = gr.Textbox(label=\"Processing Output\")
    process_btn.click(process_files, outputs=process_output)

demo.launch(share=True)
```

---

## 🧠 **How to Extend the Project**

You can replace the “dummy processing” block with:

* OCR pipeline
* Log parsing
* Text extraction
* Data cleaning
* Clustering / vectorization
* ML inference

Just insert your code inside:

```python
# PLACE YOUR PROCESSING LOGIC HERE
```

# B-MOD Tri-OCR Demo

This project provides an interactive OCR demo using **three engines**:
- **TrOCR** (Transformer-based OCR)
- **Tesseract OCR**
- **EasyOCR**

## Features
- Upload a document image and get OCR outputs from all three models.
- Compare results side by side.
- Works on CPU (GPU optional if available).

## How to run locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python app.py` (or in a Jupyter/Colab notebook)
3. Upload an image to see the outputs.

## Notes
- TrOCR model weights are loaded automatically from Hugging Face Hub.
- No need for Google Drive or local datasets.
- Designed for demo and evaluation purposes. For large-scale OCR training, use datasets with ground-truth labels.

---
---

## 🐞 **Challenges We Encountered**

During development on Hugging Face:

* GPU is not available → forced to run CPU-only
* Some sessions restarted suddenly
* Uploading large files can timeout
* Processing must be lightweight to avoid memory kills
* Long-running ML models cannot be deployed on the free tier

This version is built around those constraints.

---

## 🌐 **Deployment**

The app is ready to be deployed on Hugging Face Spaces using:

```
python: >=3.10
gradio: >=4.0
```

---

## 💬 **Credits**

Developed by **NauRaa** with guidance and iteration.


---
