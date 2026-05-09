"""
PaddleOCR Test - STABLE VERSION that works on Windows
Using version 2.8.1 which doesn't have the PIR attribute bug
"""

import sys
import os
import pandas as pd
from PIL import Image
import io
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🏥 PADDLEOCR STABLE VERSION - MEDICAL PRESCRIPTIONS")
print("=" * 70)

# Set environment variable to avoid unnecessary logs
os.environ['FLAGS_use_cuda'] = 'False'

# Load dataset
print("\n📚 Loading dataset...")
df = pd.read_parquet("hf://datasets/chaithanyakota/100-handwritten-medical-records/data/train-00000-of-00001.parquet")
print(f"✅ Loaded {len(df)} records")

# Initialize PaddleOCR with stable configuration
print("\n🔧 Initializing PaddleOCR (stable version)...")

try:
    from paddleocr import PaddleOCR

    # Use minimal initialization that worked in older versions
    ocr = PaddleOCR(
        use_angle_cls=True,  # Works in stable version
        lang='en',
        show_log=False,
        use_gpu=False
    )
    print("✅ PaddleOCR ready!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nSwitching to EasyOCR...")
    print("Installing EasyOCR...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'easyocr'])
    import easyocr
    ocr = easyocr.Reader(['en'], gpu=False)
    print("✅ EasyOCR ready!")
    ocr_type = "easyocr"

def extract_image(row):
    """Extract PIL image from dataset row"""
    img_data = row['image']

    if isinstance(img_data, dict) and 'bytes' in img_data:
        return Image.open(io.BytesIO(img_data['bytes']))
    elif isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data))
    elif isinstance(img_data, Image.Image):
        return img_data
    else:
        return None

# Test on first 3 records
print("\n" + "=" * 70)
print("📝 OCR RESULTS")
print("=" * 70)

results = []

for idx in range(min(3, len(df))):
    print(f"\n📄 RECORD {idx + 1}")

    try:
        img = extract_image(df.iloc[idx])
        if img is None:
            print("  ❌ Could not extract image")
            continue

        print(f"  Image size: {img.size}")

        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_np = np.array(img)

        # Run OCR based on which engine is available
        print("  Running OCR...")

        if 'easyocr' in str(type(ocr)):
            # EasyOCR
            result = ocr.readtext(img_np, paragraph=True)

            if result:
                print(f"  ✅ Text detected!")
                all_text = []
                for bbox, text, confidence in result:
                    all_text.append(text)
                    print(f"     - {text[:80]}... (conf: {confidence:.2f})")

                full_text = '\n'.join(all_text)
                results.append({'record': idx, 'text': full_text, 'success': True})
            else:
                print("  ⚠️ No text detected")
                results.append({'record': idx, 'text': '', 'success': False})

        else:
            # PaddleOCR
            result = ocr.ocr(img_np, cls=True)

            if result and result[0]:
                print(f"  ✅ Text detected!")
                all_text = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                        conf = line[1][1] if isinstance(line[1], tuple) and len(line[1]) > 1 else 0.5
                        all_text.append(text)
                        print(f"     - {text[:80]}... (conf: {conf:.2f})")

                full_text = '\n'.join(all_text)
                results.append({'record': idx, 'text': full_text, 'success': True})
            else:
                print("  ⚠️ No text detected")
                results.append({'record': idx, 'text': '', 'success': False})

        # Save results
        if results[-1]['success']:
            with open(f'ocr_record_{idx}_text.txt', 'w', encoding='utf-8') as f:
                f.write(results[-1]['text'])
            print(f"  💾 Saved: ocr_record_{idx}_text.txt")

        # Save image
        img.save(f'ocr_record_{idx}_image.png')
        print(f"  💾 Saved image: ocr_record_{idx}_image.png")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append({'record': idx, 'text': '', 'success': False})

# Summary
print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)

successful = sum(1 for r in results if r['success'])
print(f"\n✅ Successful: {successful}/{len(results)} records")

if successful > 0:
    print(f"\n📝 Sample text from first successful record:")
    for r in results:
        if r['success']:
            print("-" * 50)
            print(r['text'][:300])
            print("-" * 50)
            break

print("\n✅ OCR TEST COMPLETE!")