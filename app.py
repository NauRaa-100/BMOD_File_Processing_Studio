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
        return "❌ Please upload a ZIP file."

    # Clear previous uploads
    shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save the uploaded file
    zip_path = os.path.join(UPLOAD_DIR, "data.zip")
    zip_file.save(zip_path)

    # Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(UPLOAD_DIR)

    return "✅ ZIP extracted successfully. Files are ready for processing."

# ------------------------------
# 3) Process Files (Dummy Example)
# ------------------------------
def process_files():
    if not os.listdir(UPLOAD_DIR):
        return "❌ No uploaded files found."

    # Clear previous processed files
    shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Simple processing: copy extracted files
    shutil.copytree(UPLOAD_DIR, PROCESSED_DIR, dirs_exist_ok=True)

    return f"✅ Processing completed.\n📁 Total folders/files: {len(os.listdir(PROCESSED_DIR))}"

# ------------------------------
# 4) UI
# ------------------------------
with gr.Blocks(title="BMOD Line OCR – Data Processor") as demo:
    gr.Markdown("""
    # 📄 **BMOD Line OCR – Data Preparation Tool**
    Upload your dataset as a **ZIP file**, and the system will extract and prepare it directly on the server.
    """)

    with gr.Row():
        zip_input = gr.File(label="Upload ZIP File", file_types=[".zip"])
        extract_btn = gr.Button("Extract ZIP")

    extract_output = gr.Textbox(label="Extraction Status")

    extract_btn.click(
        extract_zip,
        inputs=zip_input,
        outputs=extract_output
    )

    process_btn = gr.Button("Run Processing")
    process_output = gr.Textbox(label="Processing Status")

    process_btn.click(
        process_files,
        outputs=process_output
    )

demo.launch(share=True)
