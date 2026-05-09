"""
FIXED: 4-Way OCR Comparison on a Single Image
Engines: Tesseract, PaddleOCR, EasyOCR, TrOCR (Handwritten)
"""

import sys
import time
import re
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import torch

print("=" * 80)
print("🏥 FINAL 4-WAY OCR COMPARISON (Tesseract vs PaddleOCR vs EasyOCR vs TrOCR)")
print("=" * 80)

# ============ 1. GET IMAGE PATH ============
if len(sys.argv) < 2:
    print("\n❌ Please provide an image path!")
    print("Usage: python fixed_ocr_comparison.py image.jpg")
    print("\n💡 Tip: Place your image in this folder, or provide the full path.")
    sys.exit(1)

image_path = Path(sys.argv[1])
if not image_path.exists():
    print(f"\n❌ Image not found: {image_path}")
    sys.exit(1)

print(f"\n✅ Using image: {image_path.name}")

# ============ 2. LOAD AND PREPROCESS IMAGE ============
print("\n📸 Loading and preprocessing image...")
img = Image.open(image_path).convert("RGB")
img_enhanced = ImageEnhance.Contrast(img).enhance(1.8)
img_enhanced.save("preprocessed_image.png")
print(f"   Size: {img.size}")
print(f"   Saved preprocessed version: preprocessed_image.png")

# ============ 3. INITIALIZE OCR ENGINES ============
print("\n🔧 Loading OCR engines (this may take a moment for TrOCR)...")

# Initialize flags
tesseract_ok = False
paddle_ok = False
easy_ok = False
trocr_ok = False

# Dictionary to store results
results = {}

# --- 1. Tesseract ---
try:
    import pytesseract
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in tesseract_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_ok = True
            print("✅ Tesseract ready")
            break
    if not tesseract_ok:
        print("⚠️ Tesseract not found in common paths. Skipping.")
except ImportError:
    print("⚠️ pytesseract not installed. Skipping.")
except Exception as e:
    print(f"⚠️ Tesseract error: {e}")

# --- 2. PaddleOCR ---
try:
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    paddle_ok = True
    print("✅ PaddleOCR ready")
except ImportError:
    print("⚠️ PaddleOCR not installed. Skipping.")
except Exception as e:
    print(f"⚠️ PaddleOCR error: {str(e)[:100]}")

# --- 3. EasyOCR ---
try:
    import easyocr
    easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    easy_ok = True
    print("✅ EasyOCR ready")
except ImportError:
    print("⚠️ EasyOCR not installed. Skipping.")
except Exception as e:
    print(f"⚠️ EasyOCR error: {str(e)[:100]}")

# --- 4. TrOCR (Handwritten) ---
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
    trocr_ok = True
    print(f"✅ TrOCR ready on {device}")
except ImportError:
    print("⚠️ Transformers not installed. Install with: pip install transformers")
except Exception as e:
    print(f"⚠️ TrOCR error: {str(e)[:100]}")

# ============ 4. OCR FUNCTIONS ============
def run_tesseract(img):
    if not tesseract_ok:
        return "", 0
    try:
        start = time.time()
        gray = img.convert('L')
        text = pytesseract.image_to_string(gray)
        elapsed = time.time() - start
        return text.strip(), elapsed
    except Exception as e:
        return f"Error: {e}", 0

def run_paddle(img):
    if not paddle_ok:
        return "", 0
    try:
        start = time.time()
        img_np = np.array(img)
        result = paddle_ocr.ocr(img_np, cls=True)
        elapsed = time.time() - start

        if result and result[0]:
            texts = []
            for line in result[0]:
                if line and len(line) >= 2:
                    if isinstance(line[1], tuple):
                        texts.append(line[1][0])
                    else:
                        texts.append(str(line[1]))
            return '\n'.join(texts), elapsed
        return "", elapsed
    except Exception as e:
        return f"Error: {e}", 0

def run_easy(img):
    if not easy_ok:
        return "", 0
    try:
        start = time.time()
        img_np = np.array(img)
        result = easy_reader.readtext(img_np, paragraph=True)
        elapsed = time.time() - start

        if result:
            texts = [item[1] for item in result]
            return '\n'.join(texts), elapsed
        return "", elapsed
    except Exception as e:
        return f"Error: {e}", 0

def run_trocr(img):
    if not trocr_ok:
        return "", 0
    try:
        start = time.time()
        pixel_values = trocr_processor(img, return_tensors="pt").pixel_values.to(device)
        generated_ids = trocr_model.generate(pixel_values, max_length=512)
        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elapsed = time.time() - start
        return text.strip(), elapsed
    except Exception as e:
        return f"Error: {e}", 0

# ============ 5. RUN ALL ENGINES ============
print("\n" + "=" * 80)
print("📝 RUNNING ALL OCR ENGINES...")
print("=" * 80)

# Tesseract
print("\n🔹 TESSERACT:")
text, elapsed = run_tesseract(img_enhanced)
print(f"   Time: {elapsed:.2f}s | Length: {len(text)} chars")
if text and len(text) > 0 and not text.startswith("Error"):
    print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
else:
    print(f"   Status: {text if text else 'No text detected'}")
results['Tesseract'] = {'text': text, 'time': elapsed, 'length': len(text) if text and not text.startswith("Error") else 0}

# PaddleOCR
print("\n🔹 PADDLEOCR:")
text, elapsed = run_paddle(img_enhanced)
print(f"   Time: {elapsed:.2f}s | Length: {len(text)} chars")
if text and len(text) > 0 and not text.startswith("Error"):
    print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
else:
    print(f"   Status: {text if text else 'No text detected'}")
results['PaddleOCR'] = {'text': text, 'time': elapsed, 'length': len(text) if text and not text.startswith("Error") else 0}

# EasyOCR
print("\n🔹 EASYOCR:")
text, elapsed = run_easy(img_enhanced)
print(f"   Time: {elapsed:.2f}s | Length: {len(text)} chars")
if text and len(text) > 0 and not text.startswith("Error"):
    print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
else:
    print(f"   Status: {text if text else 'No text detected'}")
results['EasyOCR'] = {'text': text, 'time': elapsed, 'length': len(text) if text and not text.startswith("Error") else 0}

# TrOCR
print("\n🔹 TROCR (Handwritten):")
text, elapsed = run_trocr(img_enhanced)
print(f"   Time: {elapsed:.2f}s | Length: {len(text)} chars")
if text and len(text) > 0 and not text.startswith("Error"):
    print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
else:
    print(f"   Status: {text if text else 'No text detected'}")
results['TrOCR'] = {'text': text, 'time': elapsed, 'length': len(text) if text and not text.startswith("Error") else 0}

# ============ 6. RESULTS TABLE ============
print("\n" + "=" * 80)
print("📊 COMPARISON RESULTS")
print("=" * 80)

print("\n{:<12} {:>10} {:>12} {}".format("Engine", "Time(s)", "Length", "Preview"))
print("-" * 70)

for engine, data in results.items():
    if data['text'] and not data['text'].startswith("Error"):
        preview = data['text'].replace('\n', ' ')[:40] + "..." if len(data['text']) > 40 else data['text']
        print("{:<12} {:>10.2f} {:>12} {}".format(
            engine, data['time'], data['length'], preview
        ))
    else:
        print("{:<12} {:>10.2f} {:>12} {}".format(
            engine, data['time'], 0, "[FAILED]"
        ))

# Determine winner
valid_results = {k: v for k, v in results.items() if v['length'] > 0 and not v['text'].startswith("Error")}
if valid_results:
    best_length = max(valid_results.items(), key=lambda x: x[1]['length'])
    print("\n" + "-" * 70)
    print(f"🏆 WINNER (Most Text): {best_length[0]} with {best_length[1]['length']} characters")

    # Also find fastest
    fastest = min(valid_results.items(), key=lambda x: x[1]['time'])
    print(f"⚡ FASTEST: {fastest[0]} ({fastest[1]['time']:.2f}s)")
else:
    print("\n⚠️ No OCR engine successfully extracted text")

# ============ 7. SAVE RESULTS ============
print("\n💾 Saving full results to 'ocr_comparison_results.txt'...")
with open('ocr_comparison_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write(f"OCR COMPARISON RESULTS FOR: {image_path.name}\n")
    f.write("=" * 80 + "\n\n")

    for engine, data in results.items():
        f.write(f"\n{'='*80}\n")
        f.write(f"{engine.upper()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Time: {data['time']:.2f}s | Length: {data['length']} characters\n\n")

        if data['text'] and not data['text'].startswith("Error"):
            f.write("EXTRACTED TEXT:\n")
            f.write("-" * 50 + "\n")
            f.write(data['text'] + "\n")
        else:
            f.write(f"STATUS: {data['text'] if data['text'] else 'No text detected'}\n")

print("\n✅ COMPLETE! Check 'ocr_comparison_results.txt' for full details.")