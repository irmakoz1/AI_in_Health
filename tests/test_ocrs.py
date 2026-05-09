"""
Complete OCR Comparison: Tesseract vs PaddleOCR vs EasyOCR
Tests all three on medical prescription images
"""

import pandas as pd
from PIL import Image, ImageEnhance
import io
import numpy as np
import cv2
import time
import subprocess
import sys
from pathlib import Path

print("=" * 80)
print("🏥 OCR COMPARISON - MEDICAL PRESCRIPTIONS")
print("Tesseract vs PaddleOCR vs EasyOCR")
print("=" * 80)

# ============ 1. LOAD DATASET ============
print("\n📚 Loading dataset...")
df = pd.read_parquet("hf://datasets/chaithanyakota/100-handwritten-medical-records/data/train-00000-of-00001.parquet")
print(f"✅ Loaded {len(df)} records")

def extract_image(row):
    """Extract PIL image from dataset row"""
    img_data = row['image']
    if isinstance(img_data, dict) and 'bytes' in img_data:
        return Image.open(io.BytesIO(img_data['bytes']))
    elif isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data))
    elif isinstance(img_data, Image.Image):
        return img_data
    return None

# ============ 2. SETUP TESSERACT ============
print("\n🔧 Setting up Tesseract...")
try:
    import pytesseract
    # Try to find tesseract path
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    tesseract_found = False
    for path in tesseract_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            print(f"✅ Tesseract found at: {path}")
            break
    if not tesseract_found:
        print("⚠️ Tesseract not found - will skip")
    tesseract_available = tesseract_found
except ImportError:
    print("⚠️ pytesseract not installed - run: pip install pytesseract")
    tesseract_available = False

# ============ 3. SETUP PADDLEOCR ============
print("\n🔧 Setting up PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    print("✅ PaddleOCR ready")
    paddle_available = True
except Exception as e:
    print(f"⚠️ PaddleOCR not available: {e}")
    paddle_available = False

# ============ 4. SETUP EASYOCR ============
print("\n🔧 Setting up EasyOCR...")
try:
    import easyocr
    easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("✅ EasyOCR ready")
    easy_available = True
except Exception as e:
    print(f"⚠️ EasyOCR not available: {e}")
    easy_available = False

# ============ 5. OCR FUNCTIONS ============
def ocr_tesseract(img):
    """Run Tesseract OCR"""
    if not tesseract_available:
        return "", 0
    try:
        start = time.time()
        # Preprocess for better results
        img_gray = img.convert('L')
        enhancer = ImageEnhance.Contrast(img_gray)
        img_enhanced = enhancer.enhance(2.0)
        text = pytesseract.image_to_string(img_enhanced)
        elapsed = time.time() - start
        return text.strip(), elapsed
    except Exception as e:
        return f"Error: {e}", 0

def ocr_paddle(img):
    """Run PaddleOCR"""
    if not paddle_available:
        return "", 0
    try:
        start = time.time()
        img_np = np.array(img)
        result = paddle_ocr.ocr(img_np, cls=True)
        elapsed = time.time() - start

        if result and result[0]:
            texts = [line[1][0] for line in result[0] if line and len(line) >= 2]
            return '\n'.join(texts), elapsed
        return "", elapsed
    except Exception as e:
        return f"Error: {e}", 0

def ocr_easy(img):
    """Run EasyOCR"""
    if not easy_available:
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

# ============ 6. RUN COMPARISON ============
print("\n" + "=" * 80)
print("📝 RUNNING COMPARISON ON FIRST 3 RECORDS")
print("=" * 80)

all_results = []

for idx in range(min(3, len(df))):
    print(f"\n{'='*80}")
    print(f"📄 RECORD {idx + 1}")
    print(f"{'='*80}")

    # Get image
    img = extract_image(df.iloc[idx])
    if img is None:
        print("❌ Could not extract image")
        continue

    print(f"Image size: {img.size}")

    # Show ground truth if available
    medicines = df.iloc[idx].get('medicines')
    if medicines and pd.notna(medicines):
        print(f"Ground truth: {str(medicines)[:100]}")

    # Results storage
    record_results = {'record': idx, 'image_size': img.size}

    # Test each OCR engine
    print("\n" + "-" * 40)
    print("OCR RESULTS:")
    print("-" * 40)

    # Tesseract
    if tesseract_available:
        print("\n🔹 TESSERACT:")
        text, elapsed = ocr_tesseract(img)
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Length: {len(text)} chars, {len(text.split())} words")
        print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
        record_results['tesseract'] = {'text': text, 'time': elapsed, 'length': len(text)}
    else:
        print("\n🔹 TESSERACT: Not available")
        record_results['tesseract'] = {'text': '', 'time': 0, 'length': 0}

    # PaddleOCR
    if paddle_available:
        print("\n🔹 PADDLEOCR:")
        text, elapsed = ocr_paddle(img)
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Length: {len(text)} chars, {len(text.split())} words")
        print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
        record_results['paddle'] = {'text': text, 'time': elapsed, 'length': len(text)}
    else:
        print("\n🔹 PADDLEOCR: Not available")
        record_results['paddle'] = {'text': '', 'time': 0, 'length': 0}

    # EasyOCR
    if easy_available:
        print("\n🔹 EASYOCR:")
        text, elapsed = ocr_easy(img)
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Length: {len(text)} chars, {len(text.split())} words")
        print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
        record_results['easy'] = {'text': text, 'time': elapsed, 'length': len(text)}
    else:
        print("\n🔹 EASYOCR: Not available")
        record_results['easy'] = {'text': '', 'time': 0, 'length': 0}

    # Save image
    img.save(f'comparison_record_{idx}.png')
    print(f"\n💾 Saved: comparison_record_{idx}.png")

    all_results.append(record_results)

# ============ 7. SUMMARY ============
print("\n" + "=" * 80)
print("📊 COMPARISON SUMMARY")
print("=" * 80)

# Create comparison table
comparison_data = []
for result in all_results:
    for engine in ['tesseract', 'paddle', 'easy']:
        if result[engine]['length'] > 0:
            comparison_data.append({
                'Record': result['record'],
                'Engine': engine.upper(),
                'Text Length': result[engine]['length'],
                'Time (s)': result[engine]['time'],
                'Has Medical Terms': any(term in result[engine]['text'].lower()
                    for term in ['mg', 'ml', 'tablet', 'patient', 'doctor', 'tsh', 'adl'])
            })

if comparison_data:
    df_results = pd.DataFrame(comparison_data)
    print("\n", df_results.to_string(index=False))

    # Best by text length
    print("\n" + "-" * 40)
    print("🏆 WINNER BY TEXT LENGTH:")
    for record in all_results:
        best = max(['tesseract', 'paddle', 'easy'],
                   key=lambda x: record[x]['length'] if record[x]['length'] > 0 else -1)
        print(f"   Record {record['record']}: {best.upper()} ({record[best]['length']} chars)")

    # Best by speed
    print("\n⚡ WINNER BY SPEED:")
    for record in all_results:
        fastest = min(['tesseract', 'paddle', 'easy'],
                      key=lambda x: record[x]['time'] if record[x]['time'] > 0 else float('inf'))
        print(f"   Record {record['record']}: {fastest.upper()} ({record[fastest]['time']:.2f}s)")
else:
    print("\n⚠️ No OCR engine produced results")

# ============ 8. RECOMMENDATION ============
print("\n" + "=" * 80)
print("💡 RECOMMENDATION")
print("=" * 80)

# Count which engine produced the most text
engine_totals = {'tesseract': 0, 'paddle': 0, 'easy': 0}
for record in all_results:
    for engine in engine_totals.keys():
        engine_totals[engine] += record[engine]['length']

if max(engine_totals.values()) > 0:
    best_engine = max(engine_totals, key=engine_totals.get)
    print(f"\n✅ Based on total text extracted ({engine_totals[best_engine]} chars),")
    print(f"   the best OCR engine for your dataset is: {best_engine.upper()}")

    if best_engine == 'tesseract':
        print("\n   Tesseract works best. Continue with Tesseract for your pipeline.")
    elif best_engine == 'paddle':
        print("\n   PaddleOCR works best. Use it for your pipeline.")
    else:
        print("\n   EasyOCR works best. Use it for your pipeline.")

print("\n" + "=" * 80)
print("✅ COMPARISON COMPLETE!")
print("=" * 80)

# Save results
with open('ocr_comparison_results.txt', 'w', encoding='utf-8') as f:
    for result in all_results:
        f.write(f"\n{'='*60}\n")
        f.write(f"RECORD {result['record']}\n")
        f.write(f"{'='*60}\n")
        for engine in ['tesseract', 'paddle', 'easy']:
            f.write(f"\n{engine.upper()}:\n")
            f.write(f"  Text: {result[engine]['text'][:500]}\n")
            f.write(f"  Length: {result[engine]['length']}\n")
            f.write(f"  Time: {result[engine]['time']:.2f}s\n")
        f.write(f"\nImage saved: comparison_record_{result['record']}.png\n")

print("\n📁 Detailed results saved to: ocr_comparison_results.txt")