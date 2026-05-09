"""
Simple Data Wrapper for Medical Handwriting Dataset
Just loads and structures the data - NO preprocessing
"""

import pandas as pd
import numpy as np
from PIL import Image
import io
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MedicalRecord:
    """Single medical record - raw data only"""
    record_id: str
    image_data: Any  # Raw image data as it comes from dataset
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_image_as_pil(self) -> Image.Image:
        """Convert raw image data to PIL Image when needed"""
        img_data = self.image_data

        if isinstance(img_data, dict) and 'bytes' in img_data:
            return Image.open(io.BytesIO(img_data['bytes']))
        elif isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data))
        elif isinstance(img_data, Image.Image):
            return img_data
        else:
            raise ValueError(f"Cannot convert {type(img_data)} to PIL Image")

    def get_image_shape(self) -> tuple:
        """Get image dimensions without loading full image when possible"""
        try:
            pil_img = self.get_image_as_pil()
            return pil_img.size
        except:
            return (0, 0)

    def preview(self) -> str:
        """Get a preview of the record"""
        return f"""
        📋 Record ID: {self.record_id}
        📊 Metadata fields: {list(self.metadata.keys())}
        🖼️  Image type: {type(self.image_data).__name__}
        """


class MedicalDatasetWrapper:
    """
    Simple wrapper that just loads and structures the data
    NO preprocessing, NO conversion, NO resizing
    """

    def __init__(self,
                 dataset_path: str = "hf://datasets/chaithanyakota/100-handwritten-medical-records/data/train-00000-of-00001.parquet"):
        """
        Initialize and load the dataset

        Args:
            dataset_path: Path to the parquet file
        """
        self.dataset_path = dataset_path
        self.df = None
        self._current_index = 0

        # Load the dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load the parquet dataset"""
        print(f"📚 Loading dataset from: {self.dataset_path}")
        self.df = pd.read_parquet(self.dataset_path)
        print(f"✅ Loaded {len(self.df)} medical records")
        print(f"📊 Columns: {self.df.columns.tolist()}")

        # Detect image column
        self.image_column = self._detect_image_column()
        print(f"🖼️  Image column: {self.image_column}")

        # Add record IDs if not present
        if 'id' not in self.df.columns and 'record_id' not in self.df.columns:
            self.df['record_id'] = [f"REC_{i:04d}" for i in range(len(self.df))]
            self.id_column = 'record_id'
        elif 'id' in self.df.columns:
            self.id_column = 'id'
        else:
            self.id_column = 'record_id'

    def _detect_image_column(self) -> str:
        """Detect which column contains image data"""
        for col in self.df.columns:
            col_lower = col.lower()
            if 'image' in col_lower or 'img' in col_lower or 'picture' in col_lower:
                return col
        # Default to first column if no image column found
        return self.df.columns[0]

    def get_record(self, index: int) -> MedicalRecord:
        """
        Get a single medical record by index - RAW DATA ONLY

        Args:
            index: Row index

        Returns:
            MedicalRecord object with raw data
        """
        if index < 0 or index >= len(self.df):
            raise IndexError(f"Index {index} out of range (0-{len(self.df)-1})")

        row = self.df.iloc[index]

        # Get raw image data (as is, no conversion)
        image_data = row[self.image_column]

        # Extract metadata (everything except the image column)
        metadata = {}
        for col in self.df.columns:
            if col != self.image_column:
                value = row[col]
                # Handle pandas NA values
                if pd.isna(value):
                    value = None
                # Convert numpy types to Python types for JSON serialization
                elif hasattr(value, 'tolist'):
                    value = value.tolist()
                metadata[col] = value

        # Get record ID
        record_id = str(row.get(self.id_column, f"REC_{index:04d}"))

        # Create record
        record = MedicalRecord(
            record_id=record_id,
            image_data=image_data,
            metadata=metadata
        )

        return record

    def get_batch(self, start_idx: int, batch_size: int) -> List[MedicalRecord]:
        """
        Get a batch of records

        Args:
            start_idx: Starting index
            batch_size: Number of records to get

        Returns:
            List of MedicalRecord objects
        """
        end_idx = min(start_idx + batch_size, len(self.df))
        records = []

        for i in range(start_idx, end_idx):
            try:
                record = self.get_record(i)
                records.append(record)
            except Exception as e:
                print(f"⚠️ Error loading record {i}: {e}")
                continue

        return records

    def iterate(self) -> Generator[MedicalRecord, None, None]:
        """
        Iterate through all records
        """
        for i in range(len(self.df)):
            yield self.get_record(i)

    def get_sample(self, num_samples: int = 5, random: bool = False) -> List[MedicalRecord]:
        """
        Get sample records for testing

        Args:
            num_samples: Number of samples to get
            random: Whether to sample randomly

        Returns:
            List of MedicalRecord objects
        """
        if random:
            indices = np.random.choice(len(self.df), min(num_samples, len(self.df)), replace=False)
        else:
            indices = range(min(num_samples, len(self.df)))

        return [self.get_record(i) for i in indices]

    def get_statistics(self) -> Dict:
        """
        Get basic dataset statistics
        """
        stats = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'image_column': self.image_column,
            'id_column': self.id_column,
            'missing_values': self.df.isnull().sum().to_dict()
        }

        # Add data types
        stats['dtypes'] = {col: str(dtype) for col, dtype in self.df.dtypes.items()}

        return stats

    def info(self):
        """Print dataset information"""
        print("\n" + "=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        print(f"Total records: {len(self.df)}")
        print(f"Total columns: {len(self.df.columns)}")
        print(f"\nColumns:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            nulls = self.df[col].isnull().sum()
            print(f"  • {col}: {dtype} (nulls: {nulls})")
        print(f"\nImage column: {self.image_column}")
        print("=" * 50)

    def __len__(self) -> int:
        """Number of records in dataset"""
        return len(self.df)

    def __getitem__(self, index) -> MedicalRecord:
        """Get record by index"""
        return self.get_record(index)

    def __iter__(self):
        """Iterator over records"""
        self._current_index = 0
        return self

    def __next__(self) -> MedicalRecord:
        """Next record in iteration"""
        if self._current_index >= len(self.df):
            raise StopIteration
        record = self.get_record(self._current_index)
        self._current_index += 1
        return record


# ============ Quick Access ============

def load_dataset() -> MedicalDatasetWrapper:
    """Quick function to load the dataset"""
    return MedicalDatasetWrapper()


# ============ Demo ============

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 Medical Dataset Wrapper - Simple Version")
    print("=" * 60)

    # Load dataset
    print("\n📚 Loading dataset...")
    dataset = load_dataset()

    # Show info
    dataset.info()

    # Get first record
    print("\n📋 First record:")
    first_record = dataset[0]
    print(f"  ID: {first_record.record_id}")
    print(f"  Image data type: {type(first_record.image_data)}")
    print(f"  Metadata: {first_record.metadata}")

    # Get sample records
    print("\n📊 Sample records:")
    samples = dataset.get_sample(3)
    for i, record in enumerate(samples):
        print(f"  {i+1}. {record.record_id} - Image type: {type(record.image_data).__name__}")

    # Iterate through first 3 records
    print("\n🔄 Iterating through first 3 records:")
    for i, record in enumerate(dataset):
        if i >= 3:
            break
        print(f"  Record {i+1}: {record.record_id}")

    print("\n✅ Data wrapper ready!")
    print("\n💡 Usage:")
    print("  dataset = MedicalDatasetWrapper()")
    print("  record = dataset[0]  # Get first record")
    print("  pil_image = record.get_image_as_pil()  # Convert to PIL when needed")
    print("  for record in dataset:  # Iterate through all")
    print("      print(record.record_id)")