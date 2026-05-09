"""
Simple EDA using the data wrapper
"""

import sys
import os
from pathlib import Path
root_dir = Path(__file__).parent.parent  # Goes from agents/ to root/
sys.path.insert(0, str(root_dir))

# Also add current directory
sys.path.insert(0, str(Path(__file__).parent))

print(f" Python path includes:")
print(f"   - {root_dir} (root directory)")
print(f"   - {Path(__file__).parent} (agents directory)")
from data_wrapper import MedicalDatasetWrapper
from PIL import Image
import matplotlib.pyplot as plt

# Load dataset
print("Loading dataset...")
dataset = MedicalDatasetWrapper()

# Print basic info
dataset.info()

# Analyze first few records
print("\n" + "=" * 50)
print("SAMPLE RECORDS ANALYSIS")
print("=" * 50)

for i in range(min(5, len(dataset))):
    record = dataset[i]
    print(f"\nRecord {i+1}: {record.record_id}")
    print(f"  Metadata: {record.metadata}")

    # Get image info without heavy conversion
    try:
        img = record.get_image_as_pil()
        print(f"  Image size: {img.size}")
        print(f"  Image mode: {img.mode}")
    except Exception as e:
        print(f"  Image error: {e}")

# Show first image if possible
if len(dataset) > 0:
    print("\n" + "=" * 50)
    print("DISPLAYING FIRST IMAGE")
    print("=" * 50)

    record = dataset[0]
    try:
        img = record.get_image_as_pil()
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Record: {record.record_id}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Cannot display image: {e}")


        # Check what's in the medicines column
print("=" * 60)
print("🔍 ANALYZING 'medicines' COLUMN")
print("=" * 60)

# Get the raw dataframe
df = dataset.df

print(f"\n📊 Total records: {len(df)}")
print(f"❌ Records with NULL medicines: {df['medicines'].isnull().sum()}")
print(f"✅ Records with medicine data: {df['medicines'].notnull().sum()}")

# Check the data type and content of non-null entries
print(f"\n📦 Data type of medicines column: {df['medicines'].dtype}")

# Sample what's in the medicines column
print(f"\n📝 Sample of medicines column (first 5 non-null values):")
medicines_data = df[df['medicines'].notnull()]['medicines'].head(5)
for i, med in enumerate(medicines_data):
    print(f"  {i+1}. Type: {type(med).__name__}")
    print(f"     Content: {str(med)[:200]}")
    print()

# Check records WITHOUT medicines
print(f"\n📄 Records WITHOUT medicines (first 3):")
null_records = df[df['medicines'].isnull()].head(3)
for i, (idx, row) in enumerate(null_records.iterrows()):
    print(f"  {i+1}. Record ID: {row.get('record_id', idx)}")
    print(f"     Other fields: {[col for col in df.columns if col != 'medicines'][:3]}")
    print()

# Compare records with and without medicines
print(f"\n📊 Comparison:")
print(f"  Records WITH medicines: {df['medicines'].notnull().sum()}")
print(f"  Records WITHOUT medicines: {df['medicines'].isnull().sum()}")

print("\n EDA complete!")