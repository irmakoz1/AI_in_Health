"""
Medical Prescription Pipeline - Focused on Medicine Extraction from Bad Handwriting
Steps:
1. PaddleOCR on original image
2. Claude API specifically for medicine name extraction from messy OCR
"""

import re
import json
import sys
from pathlib import Path
import os
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

# ============ 1. OCR MODULE ============

class PaddleOCRRunner:
    """Run PaddleOCR on original image"""

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
        """Extract text directly from image"""
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
                if self.verbose:
                    print("   ⚠️ No text detected")
                return ""
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error: {e}")
            return ""


# ============ 2. CLAUDE MEDICINE EXTRACTOR ============

class ClaudeMedicineExtractor:
    """
    Extract medicine names from bad handwritten OCR text using Claude
    """

    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        self.model = model

        # Common medication database for reference
        self.common_medications = [
            "Warfarin", "Aspirin", "Clopidogrel", "Atorvastatin", "Simvastatin",
            "Lisinopril", "Ramipril", "Metoprolol", "Amlodipine", "Furosemide",
            "Metformin", "Insulin", "Glipizide", "Pioglitazone", "Sitagliptin",
            "Levothyroxine", "Methimazole", "Ibuprofen", "Paracetamol", "Naproxen",
            "Diclofenac", "Tramadol", "Morphine", "Amoxicillin", "Azithromycin",
            "Ciprofloxacin", "Doxycycline", "Albuterol", "Omeprazole", "Pantoprazole",
            "Gabapentin", "Pregabalin", "Carbamazepine", "Sertraline", "Fluoxetine",
            "Prednisone", "Hydrochlorothiazide", "Losartan", "Spironolactone"
        ]

        # Claude API setup
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.claude_available = self.api_key is not None

        if not self.claude_available:
            print("   ⚠️ Claude API key not set.")
        else:
            print(f"   ✅ Claude API ready (model: {self.model})")

    def extract_medicines(self, text: str) -> Dict:
        """
        Extract medicine names from messy OCR text using Claude
        """

        print("\n" + "=" * 60)
        print("🩺 STEP 2: CLAUDE MEDICINE EXTRACTION")
        print("=" * 60)

        # Clean text
        cleaned = re.sub(r'\s+', ' ', text).strip()
        print(f"   Input text: {cleaned[:200]}...")

        medicines = []

        # Try Claude
        if self.claude_available:
            print(f"\n   🤖 Claude is analyzing the messy handwriting...")
            claude_result = self._call_claude_for_medicines(cleaned)

            if claude_result and 'medicines' in claude_result:
                for med in claude_result['medicines']:
                    medicines.append({
                        'ocr_text': med.get('ocr_text', ''),
                        'medicine_name': med.get('medicine_name', ''),
                        'dosage': med.get('dosage', ''),
                        'confidence': med.get('confidence', 0.5),
                        'reasoning': med.get('reasoning', '')
                    })

        # Print results
        print(f"\n   💊 Medicines identified:")
        if medicines:
            for m in medicines:
                print(f"\n      • OCR said: '{m['ocr_text']}'")
                print(f"        → Medicine: {m['medicine_name']}")
                if m['dosage']:
                    print(f"        → Dosage: {m['dosage']}")
                print(f"        → Confidence: {m['confidence']:.0%}")
                if m.get('reasoning'):
                    print(f"        → Reasoning: {m['reasoning'][:100]}...")
        else:
            print("      No medicines clearly identified")

        # Also extract any numbers/dosages separately
        dosages = self._extract_dosages(cleaned)
        if dosages:
            print(f"\n   💉 Additional dosages found: {', '.join(dosages)}")

        return {
            'medicines': medicines,
            'dosages': dosages,
            'claude_used': self.claude_available,
            'original_text': cleaned
        }

    def _call_claude_for_medicines(self, text: str) -> Optional[Dict]:
        """
        Call Claude API specifically for medicine extraction from bad handwriting
        """

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            # Specialized prompt for medicine extraction from bad handwriting
            prompt = f"""You are analyzing HANDWRITTEN medical prescription text from an OCR system.

The OCR output is often INCORRECT because handwriting is hard to read. Your job is to figure out what medicine the doctor ACTUALLY meant.

OCR Text (may have errors): "{text}"

Common medications include: {', '.join(self.common_medications[:30])}

For each potential medicine mentioned in the text:
1. Look at what the OCR read (might be misspelled like "Caboigtin" or "IeV")
2. Figure out what real medicine it's trying to say
3. Extract any dosage information (numbers like "5mg", "500mg", etc.)

Return ONLY valid JSON in this exact format:
{{
  "medicines": [
    {{
      "ocr_text": "what the OCR actually output",
      "medicine_name": "correct medicine name",
      "dosage": "dosage if found (e.g., '5mg')",
      "confidence": 0.85,
      "reasoning": "brief explanation (e.g., 'Caboigtin looks like Carbamazepine when handwritten')"
    }}
  ]
}}

If no medicines are found, return an empty array: {{"medicines": []}}"""

            message = client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return None

        except ImportError:
            print("   ❌ anthropic not installed. Run: pip install anthropic")
            return None
        except Exception as e:
            print(f"   ⚠️ Claude API error: {e}")
            return None

    def _extract_dosages(self, text: str) -> List[str]:
        """Extract dosage patterns from text"""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|μg|g|ml|iu)',
            r'(\d+(?:\.\d+)?)\s*(?:mg|mcg)(?![a-z])',
            r'(\d+)\s*(?:tablet|capsule|pill)',
        ]

        dosages = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    dosages.append(f"{match[0]}{match[1] if len(match) > 1 else 'mg'}")
                else:
                    dosages.append(f"{match}mg")

        return list(set(dosages))


# ============ 3. MAIN PIPELINE ============

class MedicalPrescriptionPipeline:
    """Pipeline: OCR + Claude medicine extraction"""

    def __init__(self, claude_api_key: str = None, verbose: bool = True):
        self.verbose = verbose
        self.ocr = PaddleOCRRunner(verbose)
        self.extractor = ClaudeMedicineExtractor(claude_api_key)

    def process(self, image_path: str) -> Dict:
        """Process prescription image"""

        print("\n" + "=" * 70)
        print("🏥 MEDICAL PRESCRIPTION PIPELINE")
        print(f"   Image: {Path(image_path).name}")
        print("=" * 70)

        # Step 1: OCR
        text = self.ocr.extract_text(image_path)

        if not text:
            print("\n❌ No text extracted")
            return {'success': False}

        # Step 2: Extract medicines with Claude
        medical_data = self.extractor.extract_medicines(text)

        # Final results
        result = {
            'success': True,
            'image': image_path,
            'extracted_text': medical_data['original_text'],
            'medicines': medical_data['medicines'],
            'dosages': medical_data['dosages'],
            'claude_used': medical_data['claude_used']
        }

        # Save results
        with open('medicines_extracted.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)

        if result['medicines']:
            print("\n✅ Medicines found:")
            for m in result['medicines']:
                print(f"   • {m['medicine_name']} (from OCR: '{m['ocr_text']}')")
        else:
            print("\n⚠️ No medicines clearly identified")

        print(f"\n💾 Results saved to: medicines_extracted.json")

        return result


# ============ RUN ============

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python claude_medicine_extractor.py <image_path>")
        print("\nSet your Claude API key:")
        print("  set ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    # Run pipeline
    pipeline = MedicalPrescriptionPipeline(verbose=True)
    result = pipeline.process(image_path)