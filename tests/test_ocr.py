"""
Simple OCR Test - Standalone
Directly loads the dataset and tests OCR
"""

import pandas as pd
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

# ============ 1. LOAD DATASET DIRECTLY ============
print("=" * 60)
print("📚 LOADING DATASET DIRECTLY")
print("=" * 60)

# Load the parquet file directly
df = pd.read_parquet("hf://datasets/chaithanyakota/100-handwritten-medical-records/data/train-00000-of-00001.parquet")

print(f"\n✅ Loaded {len(df)} records")
print(f"📊 Columns: {df.columns.tolist()}")

# ============ 2. EXAMINE FIRST RECORD ============
print("\n" + "=" * 60)
print("🔍 EXAMINING FIRST RECORD")
print("=" * 60)

first_row = df.iloc[0]
print(f"\nRow 0 contents:")
for col in df.columns:
    value = first_row[col]
    print(f"  {col}: {type(value).__name__}")
    if col == 'image':
        print(f"    Image type: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"    Image shape: {value.shape}")
    else:
        print(f"    Value: {str(value)[:100]}")

# ============ 3. EXTRACT AND DISPLAY IMAGE ============
print("\n" + "=" * 60)
print("🖼️ EXTRACTING IMAGE FROM FIRST RECORD")
print("=" * 60)

# Try to extract image
image_data = first_row['image']
print(f"\nImage data type: {type(image_data)}")

# Convert to PIL Image
try:
    if isinstance(image_data, dict) and 'bytes' in image_data:
        print("  Image is in dict with 'bytes' key")
        img = Image.open(io.BytesIO(image_data['bytes']))
    elif isinstance(image_data, bytes):
        print("  Image is bytes")
        img = Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        print("  Image is already PIL Image")
        img = image_data
    elif hasattr(image_data, 'to_pil'):
        print("  Image has to_pil method")
        img = image_data.to_pil()
    else:
        print(f"  Unknown image format: {type(image_data)}")
        # Try to display as array
        if hasattr(image_data, '__array__'):
            img_array = np.array(image_data)
            img = Image.fromarray(img_array)
        else:
            raise ValueError("Cannot extract image")

    print(f"  ✅ Image loaded: {img.size}, {img.mode}")

    # Display image
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.title(f"First Record Image\nSize: {img.size}, Mode: {img.mode}")
    plt.axis('off')
    plt.show()

    # Save image
    img.save("first_record_image.png")
    print(f"  💾 Saved: first_record_image.png")

except Exception as e:
    print(f"  ❌ Error extracting image: {e}")
    import traceback
    traceback.print_exc()

# ============ 4. CHECK IMAGE QUALITY ============
print("\n" + "=" * 60)
print("📊 IMAGE QUALITY ANALYSIS")
print("=" * 60)

try:
    if 'img' in locals():
        # Convert to grayscale for analysis
        gray = img.convert('L')
        img_array = np.array(gray)

        print(f"\nImage statistics:")
        print(f"  Mean brightness: {np.mean(img_array):.1f}/255")
        print(f"  Std deviation: {np.std(img_array):.1f}")
        print(f"  Min pixel: {img_array.min()}")
        print(f"  Max pixel: {img_array.max()}")

        # Check if image has text (dark pixels)
        dark_pixels = np.sum(img_array < 100)
        dark_percentage = dark_pixels / img_array.size * 100
        print(f"  Dark pixels (<100): {dark_percentage:.1f}%")

        if dark_percentage < 1:
            print(f"\n  ⚠️ WARNING: Very few dark pixels - image may be blank or very light!")
        elif dark_percentage > 30:
            print(f"\n  ⚠️ WARNING: Too many dark pixels - image may be inverted!")
        else:
            print(f"\n  ✅ Image has reasonable text density")

except Exception as e:
    print(f"Error analyzing image: {e}")

# ============ 5. CHECK OTHER RECORDS ============
print("\n" + "=" * 60)
print("📋 CHECKING FIRST 5 RECORDS")
print("=" * 60)

for i in range(min(5, len(df))):
    row = df.iloc[i]
    print(f"\nRecord {i}:")

    # Check medicines field
    if 'medicines' in df.columns:
        meds = row['medicines']
        print(f"  Medicines: {meds if meds is not None else 'NULL'}")

    # Check if image exists
    img_data = row['image']
    print(f"  Image type: {type(img_data).__name__}")

    # Quick check of image content (without full loading)
    try:
        if isinstance(img_data, dict) and 'bytes' in img_data:
            size_mb = len(img_data['bytes']) / 1024 / 1024
            print(f"  Image size: {size_mb:.2f} MB")
        elif isinstance(img_data, bytes):
            size_mb = len(img_data) / 1024 / 1024
            print(f"  Image size: {size_mb:.2f} MB")
    except:
        pass

# ============ 6. TRY SIMPLE TEXT EXTRACTION ============
print("\n" + "=" * 60)
print("🔍 ATTEMPTING SIMPLE OCR")
print("=" * 60)

try:
    import pytesseract
    from PIL import ImageEnhance

    print("\n📝 Trying Tesseract OCR on first image...")

    if 'img' in locals():
        # Enhance image
        enhancer = ImageEnhance.Contrast(img)
        enhanced = enhancer.enhance(2.0)

        # Run OCR
        text = pytesseract.image_to_string(enhanced)

        print(f"\n  Extracted text:")
        print("-" * 40)
        print(text if text.strip() else "[NO TEXT DETECTED]")
        print("-" * 40)

        if text.strip():
            print(f"\n  ✅ Tesseract found some text!")
        else:
            print(f"\n  ❌ Tesseract found no text - image may be blank")

except ImportError:
    print("\n  ⚠️ Tesseract not installed. Install with: pip install pytesseract")
    print("     Also need to install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
except Exception as e:
    print(f"\n  ❌ Error: {e}")

# ============ 7. SUMMARY ============
print("\n" + "=" * 60)
print("📊 SUMMARY & RECOMMENDATIONS")
print("=" * 60)

print("""
Based on the analysis:

1. If the image appears BLANK or all WHITE:
   → The dataset doesn't contain actual prescription images
   → Might need a different dataset

2. If the image has handwriting but OCR fails:
   → Need better preprocessing
   → Try EasyOCR instead of Tesseract
   → Install: pip install easyocr

3. If the image has printed text:
   → Standard OCR should work
   → Try different preprocessing

4. To proceed:
   → Install easyocr: pip install easyocr
   → Run: python simple_easyocr_test.py
""")

print("\n✅ Diagnostic complete!")