"""
Medical Prescription Pipeline - Fuzzy Matching (No Claude)
Works immediately without any API
"""

import re
import json
import sys
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

class MedicineExtractor:
    def __init__(self):
        # Medication patterns (OCR error -> correct name)
        self.medicine_patterns = {
            'stalevo': ['stalev', 'stal evo', 'stalevo', 'stalev0', 'stalevo12', 'STALEV', 'stalevo125'],
            'levodopa': ['levodopa', 'levo dopa', 'l-dopa', 'levodop', 'levodopa'],
            'carbidopa': ['carbidopa', 'calbidopa', 'carbidop', 'carbi dopa'],
            'entacapone': ['entacapone', 'comtan', 'entacapon'],
            'warfarin': ['warfarin', 'war farin', 'warfin'],
            'aspirin': ['aspirin', 'asprin', 'aspirn'],
            'metformin': ['metformin', 'met formin', 'metform'],
            'levothyroxine': ['levothyroxine', 'levo', 'levothyr', 'levoxyl'],
            'ibuprofen': ['ibuprofen', 'ibu profen', 'advil', 'motrin'],
            'amoxicillin': ['amoxicillin', 'amox', 'amoxicil'],
            'omeprazole': ['omeprazole', 'omeprazol', 'prilosec'],
            'gabapentin': ['gabapentin', 'gabapent', 'neurontin'],
        }

        self.medication_scores = {}

    def similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def extract_medicines(self, text: str) -> List[Dict]:
        medicines = []
        text_lower = text.lower()

        for correct_name, variants in self.medicine_patterns.items():
            for variant in variants:
                if variant in text_lower:
                    # Find the actual OCR text
                    pattern = re.compile(re.escape(variant), re.IGNORECASE)
                    match = pattern.search(text)
                    ocr_text = match.group(0) if match else variant

                    # Calculate confidence
                    confidence = 0.95 if variant == correct_name else 0.85

                    # Find dosage near this medicine
                    dosage = self._find_dosage(text, match.start() if match else 0)

                    medicines.append({
                        'medicine_name': correct_name.upper(),
                        'ocr_text': ocr_text,
                        'dosage': dosage,
                        'confidence': confidence
                    })
                    break

        # Remove duplicates
        seen = set()
        unique = []
        for m in medicines:
            if m['medicine_name'] not in seen:
                seen.add(m['medicine_name'])
                unique.append(m)

        return unique

    def _find_dosage(self, text: str, position: int) -> str:
        if position == -1:
            return ""

        start = max(0, position - 30)
        end = min(len(text), position + 50)
        surrounding = text[start:end]

        # Look for dosage patterns
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml)',
            r'/(\d+)(?:/|$)',  # For patterns like /125/312/200
            r'(\d{2,3})\s*(?:mg)?',
        ]

        for pattern in patterns:
            match = re.search(pattern, surrounding, re.IGNORECASE)
            if match:
                if pattern == r'/(\d+)(?:/|$)':
                    return match.group(0)
                elif len(match.groups()) >= 2 and match.group(2):
                    return f"{match.group(1)}{match.group(2)}"
                else:
                    return f"{match.group(1)}mg"

        return ""


# ============ OCR MODULE ============

class PaddleOCRRunner:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            if self.verbose:
                print("   ✅ PaddleOCR ready")
            self.available = True
        except ImportError:
            print("   ❌ PaddleOCR not installed. Run: pip install paddleocr")
            self.available = False

    def extract_text(self, image_path: str) -> str:
        if not self.available:
            return ""

        if self.verbose:
            print("\n" + "=" * 60)
            print("📝 STEP 1: PADDLEOCR EXTRACTION")
            print("=" * 60)

        try:
            result = self.ocr.ocr(image_path, cls=True)

            if result and result[0]:
                texts = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        if isinstance(line[1], tuple):
                            texts.append(line[1][0])
                        else:
                            texts.append(str(line[1]))

                full_text = '\n'.join(texts)
                if self.verbose:
                    print(f"   ✅ Extracted {len(full_text)} characters")
                    print(f"   Preview: {full_text[:200]}...")
                return full_text
            else:
                print("   ⚠️ No text detected")
                return ""
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return ""


# ============ MAIN PIPELINE ============

class MedicalPipeline:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.ocr = PaddleOCRRunner(verbose)
        self.extractor = MedicineExtractor()

    def process(self, image_path: str) -> Dict:
        print("\n" + "=" * 70)
        print("🏥 MEDICAL PRESCRIPTION PIPELINE")
        print(f"   Image: {Path(image_path).name}")
        print("=" * 70)

        # Step 1: OCR
        text = self.ocr.extract_text(image_path)

        if not text:
            return {'success': False, 'medicines': []}

        # Step 2: Extract medicines
        print("\n" + "=" * 60)
        print("🩺 STEP 2: MEDICINE EXTRACTION")
        print("=" * 60)

        medicines = self.extractor.extract_medicines(text)
        dosages = self._extract_all_dosages(text)

        # Print results
        print(f"\n   💊 Medicines found:")
        if medicines:
            for m in medicines:
                print(f"\n      • '{m['ocr_text']}' → {m['medicine_name']}")
                if m['dosage']:
                    print(f"        Dosage: {m['dosage']}")
                print(f"        Confidence: {m['confidence']:.0%}")
        else:
            print("      None found")

        if dosages:
            print(f"\n   💉 Additional dosages: {', '.join(dosages[:5])}")

        # Results
        result = {
            'success': True,
            'image': image_path,
            'extracted_text': text,
            'medicines': medicines,
            'dosages': dosages,
            'overall_confidence': self._calculate_confidence(medicines, text)
        }

        # Save
        with open('medicines_extracted.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        print(f"\n🎯 Overall Confidence: {result['overall_confidence']:.0%}")

        if medicines:
            print("\n✅ Medicines found:")
            for m in medicines:
                print(f"   • {m['medicine_name']}")
        else:
            print("\n⚠️ No medicines found")

        print(f"\n💾 Saved: medicines_extracted.json")

        return result

    def _extract_all_dosages(self, text: str) -> List[str]:
        dosages = []
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml)',
            r'/(\d+)(?:/|$)',
            r'(\d{2,3})(?=\s|$)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    dosages.append(f"{match[0]}{match[1] if len(match) > 1 else 'mg'}")
                else:
                    if len(str(match)) >= 2 and int(match) <= 500:
                        dosages.append(f"{match}mg")
        return list(set(dosages[:10]))

    def _calculate_confidence(self, medicines: List[Dict], text: str) -> float:
        if not medicines:
            return 0.0
        avg_conf = sum(m['confidence'] for m in medicines) / len(medicines)
        return min(1.0, avg_conf)


# ============ RUN ============

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_fuzzy_only.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    pipeline = MedicalPipeline(verbose=True)
    result = pipeline.process(image_path)