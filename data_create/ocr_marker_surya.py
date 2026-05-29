# %% [markdown]
# # Vietnamese Caselaw PDF → Markdown via Marker (Surya OCR)
#
# Uses the **Marker** pipeline (built on **Surya** VLM) to convert scanned
# Vietnamese legal PDF documents to clean markdown.
#
# Marker handles the full pipeline:
# - Text detection & OCR (Surya)
# - Layout analysis & reading order
# - Table recognition
# - Formatting & cleanup
#
# Designed for **Kaggle T4 GPU** (16 GB VRAM).

# %% Install dependencies
import subprocess, sys, os

def run(cmd):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)

run(f"{sys.executable} -m pip install -q marker-pdf")

print("\n✓ marker-pdf installed!")

# %% Imports & configuration
import gc
import time
import glob

import torch

# ── Paths ────────────────────────────────────────────────────────────────
INPUT_DIR  = "/kaggle/input/caselaw-pdfs/caselaw_pdfs"   # ← adjust to your dataset
OUTPUT_DIR = "ocr_marker"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU    : {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")

pdf_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdf")))
print(f"\nFound {len(pdf_files)} PDF(s) in {INPUT_DIR}")

# %% Load Marker pipeline (Surya models)
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

config_json = {
    "output_format": "json",
    "force_ocr": True,            # force OCR on all pages (scanned PDFs)
    "paginate_output": True,      # add page separators
    "languages": "vi",            # <--- Hint for Vietnamese OCR
    "batch_multiplier": 2,        # <--- Increase internal batch sizing on GPU (default is 1). Use 2 or 3 carefully to avoid OOM
    "max_pages": None             # Ensure it processes all pages
}
config_parser_json = ConfigParser(config_json)

config_html = {
    "output_format": "html",
    "force_ocr": config_json["force_ocr"],
    "paginate_output": config_json["paginate_output"],
    "languages": config_json["languages"],
    "batch_multiplier": config_json["batch_multiplier"],
    "max_pages": config_json["max_pages"]
}
config_parser_html = ConfigParser(config_html)

print("Loading Surya models …")
t0 = time.time()
artifact_dict = create_model_dict()

converter_json = PdfConverter(
    config=config_parser_json.generate_config_dict(),
    artifact_dict=artifact_dict,
    processor_list=config_parser_json.get_processors(),
    renderer=config_parser_json.get_renderer(),
)
converter_html = PdfConverter(
    config=config_parser_html.generate_config_dict(),
    artifact_dict=artifact_dict,
    processor_list=config_parser_html.get_processors(),
    renderer=config_parser_html.get_renderer(),
)
print(f"✓ Models loaded in {time.time() - t0:.1f}s")

if torch.cuda.is_available():
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM used: {vram:.1f} GB")

# %% Run OCR on all PDFs
print("=" * 65)
print("MARKER OCR  (Surya VLM)")
print("=" * 65)

results = []
for idx, pdf_path in enumerate(pdf_files, 1):
    fname = os.path.basename(pdf_path)
    out_path_json = os.path.join(OUTPUT_DIR, fname.replace(".pdf", ".json"))
    out_path_html = os.path.join(OUTPUT_DIR, fname.replace(".pdf", ".html"))

    # Skip if both already processed
    if os.path.exists(out_path_json) and os.path.exists(out_path_html):
        print(f"[{idx}/{len(pdf_files)}] SKIP (exists) {fname}")
        results.append({"file": fname, "status": "SKIP", "time_s": 0})
        continue

    print(f"[{idx}/{len(pdf_files)}] {fname}")
    t0 = time.time()
    try:
        # Process JSON
        rendered_json = converter_json(pdf_path)
        text_json, metadata_json, images_json = text_from_rendered(rendered_json)
        
        if text_json and text_json.strip():
            with open(out_path_json, "w", encoding="utf-8") as f:
                f.write(text_json)
            status_json = "OK"
        else:
            status_json = "EMPTY"

        # Process HTML
        rendered_html = converter_html(pdf_path)
        text_html, metadata_html, images_html = text_from_rendered(rendered_html)
        
        if text_html and text_html.strip():
            with open(out_path_html, "w", encoding="utf-8") as f:
                f.write(text_html)
            status_html = "OK"
        else:
            status_html = "EMPTY"

        dt = time.time() - t0
        
        if status_json == "OK" and status_html == "OK":
            status = "OK"
            print(f"  → OK  JSON:{len(text_json)} chars, HTML:{len(text_html)} chars ({dt:.1f}s)")
        else:
            status = f"JSON:{status_json}, HTML:{status_html}"
            print(f"  → {status} ({dt:.1f}s)")

    except Exception as e:
        dt = time.time() - t0
        status = "FAIL"
        print(f"  → FAIL: {e} ({dt:.1f}s)")

    results.append({"file": fname, "status": status, "time_s": round(dt, 1)})
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

ok = sum(1 for r in results if r["status"] == "OK")
print(f"\nMarker done: {ok}/{len(results)} OK\n")

# %% Cleanup GPU memory
del converter_json, converter_html, artifact_dict
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("GPU memory freed.")

# %% Summary report
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

ok    = sum(1 for r in results if r["status"] == "OK")
empty = sum(1 for r in results if r["status"] == "EMPTY")
fail  = sum(1 for r in results if r["status"] == "FAIL")
skip  = sum(1 for r in results if r["status"] == "SKIP")
total = len(results)
total_time = sum(r["time_s"] for r in results)

print(f"\n{'OK':>5} {'Empty':>6} {'Fail':>6} {'Skip':>6} {'Total':>6}")
print("-" * 35)
print(f"{ok:>5} {empty:>6} {fail:>6} {skip:>6} {total:>6}")
print(f"\nTotal time: {total_time/60:.1f} min")
print(f"Outputs saved to: {OUTPUT_DIR}/")

# %% Preview first extracted file
md_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
if md_files:
    with open(md_files[0], "r", encoding="utf-8") as f:
        content = f.read()
    print(f"Preview: {os.path.basename(md_files[0])}")
    print("-" * 50)
    print(content[:3000])
    if len(content) > 3000:
        print(f"\n… ({len(content) - 3000} more characters)")
