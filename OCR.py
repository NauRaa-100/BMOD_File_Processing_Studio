"""
from google.colab import drive
drive.mount('/content/drive')

!wget https://zenodo.org/record/15310982/files/b-mod_lines.zip?download=1 -O bmod_lines.zip

!unzip bmod_lines.zip -d bmod_lines

!pip install --quiet transformers datasets accelerate gradio easyocr jiwer textdistance
!apt-get update -qq && apt-get install -y -qq tesseract-ocr libtesseract-dev
!pip install --quiet pytesseract


!pip install --quiet opencv-python-headless matplotlib pillow
"""
import os
import pandas as pd
import glob

BASE_DIR = "bmod_lines"

image_paths = []
for ext in ("png", "jpg", "jpeg", "tif"):
    image_paths += glob.glob(os.path.join(BASE_DIR, f"**/*.{ext}"), recursive=True)

image_paths = sorted(image_paths)
print("Found images:", len(image_paths))

data = pd.DataFrame({"image": image_paths})

SAMPLE_MODE = False
SAMPLE_N = 200
if SAMPLE_MODE and len(data) > SAMPLE_N:
    data = data.sample(SAMPLE_N, random_state=42).reset_index(drop=True)

print(data.head())

# -----------------------------
# 1. Preprocessing functions (OpenCV)
# -----------------------------
import cv2
import numpy as np
from PIL import Image

def load_image_cv(path, as_gray=True):
    if as_gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def preprocess_image_cv(path, do_deskew=True, resize_max=1024):
    img = load_image_cv(path, as_gray=True)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > resize_max:
        scale = resize_max / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    # denoise
    img = cv2.GaussianBlur(img, (3,3), 0)
    # binarize
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # deskew
    if do_deskew:
        coords = np.column_stack(np.where(img_bin > 0))
        if coords.shape[0] > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = img_bin.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_bin = cv2.warpAffine(img_bin, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img_bin

PREPROCESSED_DIR = os.path.join("bmod_lines", "preprocessed")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

for i, row in data.head(5).iterrows():
    img_p = preprocess_image_cv(row['image'])
    if img_p is not None:
        outp = os.path.join(PREPROCESSED_DIR, os.path.basename(row['image']))
        cv2.imwrite(outp, img_p)

print("Preprocessing sample done. Saved to:", PREPROCESSED_DIR)

# -----------------------------
# 2 TrOCR, Tesseract, EasyOCR
# -----------------------------
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

print("Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.to(DEVICE)

# Tesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# EasyOCR
import easyocr
reader_easy = easyocr.Reader(['en'], gpu=(DEVICE=="cuda"))

# -----------------------------
# -----------------------------
def ocr_trocr_path(path):
    try:
        image = Image.open(path).convert("RGB")
    except:
        return ""
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    generated_ids = model.generate(pixel_values, max_length=512)
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return preds[0]

def ocr_tesseract_path(path, lang='eng'):
    try:
        img = preprocess_image_cv(path, do_deskew=True)
        if img is None:
            return ""
        pil = Image.fromarray(img)
        txt = pytesseract.image_to_string(pil, lang=lang)
        return txt.strip()
    except:
        return ""

def ocr_easyocr_path(path, lang_list=None):
    try:
        if lang_list is None:
            lang_list = ['en']
        result = reader_easy.readtext(path, detail=0)
        return " ".join(result).strip()
    except:
        return ""

# -----------------------------
# -----------------------------
SAMPLE_RUN_N = 100
sample_paths = data['image'].tolist()[:SAMPLE_RUN_N]

results = []
for p in sample_paths:
    tro = ocr_trocr_path(p)
    tes = ocr_tesseract_path(p)
    eas = ocr_easyocr_path(p)
    results.append({"image": p, "trocr": tro, "tesseract": tes, "easyocr": eas})

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(PREPROCESSED_DIR, "ocr_sample_results.csv"), index=False)
print("Saved sample results to", os.path.join(PREPROCESSED_DIR, "ocr_sample_results.csv"))

# -----------------------------
# -----------------------------
from jiwer import cer
import textdistance

def safe_norm_lev(a, b):
    if a is None or b is None:
        return 1.0
    return textdistance.levenshtein.normalized_distance(a, b)

results_df['tro_tes_cer'] = results_df.apply(lambda r: cer(r['tesseract'], r['trocr']) if r['tesseract'] and r['trocr'] else None, axis=1)
results_df['tro_eas_cer'] = results_df.apply(lambda r: cer(r['easyocr'], r['trocr']) if r['easyocr'] and r['trocr'] else None, axis=1)
results_df['lev_tro_tes'] = results_df.apply(lambda r: safe_norm_lev(r['trocr'], r['tesseract']), axis=1)

print("Mean tro_tes_cer:", results_df['tro_tes_cer'].dropna().mean())
print("Mean lev_tro_tes:", results_df['lev_tro_tes'].dropna().mean())

results_df.to_csv(os.path.join(PREPROCESSED_DIR, "ocr_sample_results_with_metrics.csv"), index=False)

# -----------------------------
# -----------------------------
import matplotlib.pyplot as plt

sample_display = results_df.sample(6)
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.flatten()
for ax, (_, row) in zip(axes, sample_display.iterrows()):
    img = Image.open(row['image']).convert('RGB')
    ax.imshow(img)
    ax.axis('off')
    title = f"TroCR:\n{row['trocr'][:80]}...\nTess:\n{row['tesseract'][:80]}..."
    ax.set_title(title, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PREPROCESSED_DIR, "sample_visualization.png"))
print("Saved sample visualization to", os.path.join(PREPROCESSED_DIR, "sample_visualization.png"))

!rm -rf OCR
!git clone https://huggingface.co/spaces/NauRaa/OCR
%cd OCR

import os
PREPROCESSED_DIR = os.path.join("bmod_lines", "preprocessed")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)


!ls bmod_lines

for i, row in data.head(5).iterrows():
    img_p = preprocess_image_cv(row['image'])
    if img_p is not None:
        outp = os.path.join(PREPROCESSED_DIR, os.path.basename(row['image']))
        cv2.imwrite(outp, img_p)

print("Preprocessed images saved to:", PREPROCESSED_DIR)


import shutil
shutil.copytree(PREPROCESSED_DIR, "./preprocessed", dirs_exist_ok=True)

!git config --global user.email "nauraa1009@gmail.com"
!git config --global user.name "NauRaa"

!huggingface-cli login

!git config --global credential.helper store

from huggingface_hub import Repository

repo_local = "./OCR"

repo = Repository(local_dir=repo_local, clone_from="https://huggingface.co/spaces/NauRaa/OCR", use_auth_token="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")

!git add .
!git commit -m "Add BMOD OCR project files"
!git push

!rm -rf OCR
!git clone https://huggingface.co/spaces/NauRaa/OCR
%cd OCR

import shutil
shutil.copytree("/content/bmod_lines/preprocessed", "./preprocessed", dirs_exist_ok=True)
shutil.copy("/content/bmod_lines/README_BMOD_PROJECT.txt", "./README_BMOD_PROJECT.txt")

README = f"""
B-MOD OCR Project
===================

Folder: /content/bmod_lines

What we did:
- Downloaded and unzipped B-MOD
- Built a DataFrame of image paths
- Implemented preprocessing (denoise, binarize, deskew)
- Ran three OCR engines (TrOCR, Tesseract, EasyOCR)
- Saved results and metrics (agreement-based CER)
- Built a Gradio demo to compare outputs interactively

Notes:
- B-MOD does not include ground-truth transcriptions; therefore we measured agreement among models instead of absolute accuracy.
- For real supervised training, use datasets with GT (e.g., IAM, FUNSD, SROIE, RVL-CDIP with OCR labels).

How to run:
1) Mount Drive
2) Ensure enough storage
3) Run the script cells in order
4) Use iface.launch(share=True) to open Gradio demo

"""
with open("/content/bmod_lines/README_BMOD_PROJECT.txt", "w") as f:
    f.write(README)

import shutil
shutil.copytree("/content/bmod_lines/preprocessed", "./preprocessed", dirs_exist_ok=True)
shutil.copy("/content/bmod_lines/README_BMOD_PROJECT.txt", "./README_BMOD_PROJECT.txt")
