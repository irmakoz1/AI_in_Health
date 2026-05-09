"""
Medical Prescription Pipeline with Claude Opus 4.7
Fixed for Claude Opus 4.7 API
"""

import os
import re
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional

# ============ FIND .env FILE IN ROOT ============
script_path = Path(__file__).resolve()
tests_dir = script_path.parent
project_root = tests_dir.parent

env_path = project_root / '.env'

if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}")
else:
    print(f"❌ .env not found at: {env_path}")
    sys.exit(1)

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    print("❌ No API key found in .env")
    sys.exit(1)

print(f"✅ API key loaded (ending: ...{ANTHROPIC_API_KEY[-8:]})")

# ============ PADDLEOCR MODULE ============

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
                    print(f"   Preview: {full_text[:300]}...")
                return full_text
            else:
                print("   ⚠️ No text detected")
                return ""
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return ""


# ============ CLAUDE EXTRACTOR ============

class ClaudeExtractor:
    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = "claude-opus-4-7"
            print(f"   ✅ Claude Opus 4.7 ready")
            self.available = True
        except Exception as e:
            print(f"   ❌ Claude error: {e}")
            self.available = False

    def extract_medicines(self, text: str) -> List[Dict]:
        if not self.available:
            return []

        print("\n   🤖 Claude Opus 4.7 analyzing prescription...")

        # Removed temperature parameter - not supported in Opus 4.7
        prompt = f"""Extract medications from this OCR text from a handwritten prescription.

OCR Text:
{text}

Common OCR errors to watch for:
- "STALEV0125/312S/200" means STALEVO
- "CALBIDOPA" means CARBIDOPA

Return ONLY valid JSON:
{{"medicines": [
  {{"ocr_text": "original text", "medicine_name": "correct name", "dosage": "dosage if found", "confidence": 0.95}}
]}}

If no medicines, return: {{"medicines": []}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                # temperature parameter removed - not supported
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            print(f"   📝 Claude response received")

            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                data = json.loads(json_match.group(0))
                medicines = data.get('medicines', [])
                print(f"   ✅ Found {len(medicines)} medicines")

                for m in medicines:
                    print(f"      - {m.get('medicine_name', '?')} (from: {m.get('ocr_text', '?')})")

                return medicines
            else:
                print(f"   ⚠️ Could not parse JSON")
                print(f"   Response: {response_text[:200]}")
                return []

        except Exception as e:
            print(f"   ❌ Claude error: {e}")
            return []


# ============ FUZZY FALLBACK ============

class FuzzyExtractor:
    @staticmethod
    def extract_medicines(text: str) -> List[Dict]:
        medicines = []
        text_lower = text.lower()

        patterns = {
            'stalevo': ['stalev', 'stal evo', 'stalevo', 'stalev0', 'stalevo12'],
            'levodopa': ['levodopa', 'levo dopa', 'l-dopa', 'levodop'],
            'carbidopa': ['carbidopa', 'calbidopa', 'carbidop'],
        }

        for correct_name, variants in patterns.items():
            for variant in variants:
                if variant in text_lower:
                    medicines.append({
                        'ocr_text': variant,
                        'medicine_name': correct_name.upper(),
                        'dosage': FuzzyExtractor._find_dosage(text, text.find(variant)),
                        'confidence': 0.85
                    })
                    break

        return medicines

    @staticmethod
    def _find_dosage(text: str, position: int) -> str:
        if position == -1:
            return ""
        start = max(0, position - 20)
        end = min(len(text), position + 40)
        surrounding = text[start:end]

        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml)', surrounding, re.IGNORECASE)
        if match:
            return match.group(0)

        slash_match = re.search(r'(/\d+)+', surrounding)
        if slash_match:
            return slash_match.group(0)

        return ""


# ============ MAIN PIPELINE ============

class MedicalPipeline:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.ocr = PaddleOCRRunner(verbose)
        self.claude = ClaudeExtractor()
        self.fuzzy = FuzzyExtractor()

    def process(self, image_path: str) -> Dict:
        print("\n" + "=" * 70)
        print("🏥 MEDICAL PRESCRIPTION PIPELINE")
        print(f"   Model: Claude Opus 4.7")
        print(f"   Image: {Path(image_path).name}")
        print("=" * 70)

        # Step 1: OCR
        text = self.ocr.extract_text(image_path)

        if not text:
            print("\n❌ No text extracted")
            return {'success': False, 'medicines': []}

        # Step 2: Extract medicines with Claude
        print("\n" + "=" * 60)
        print("🩺 STEP 2: MEDICINE EXTRACTION")
        print("=" * 60)

        medicines = self.claude.extract_medicines(text)

        # Fallback to fuzzy if Claude found nothing
        if not medicines:
            print("\n   📊 Claude found nothing, trying fuzzy matching...")
            medicines = self.fuzzy.extract_medicines(text)

        # Print results
        print(f"\n   💊 Final medicines found: {len(medicines)}")
        for m in medicines:
            print(f"\n      • OCR: '{m.get('ocr_text', '')}'")
            print(f"        → Medicine: {m.get('medicine_name', '')}")
            if m.get('dosage'):
                print(f"        → Dosage: {m['dosage']}")
            print(f"        → Confidence: {m.get('confidence', 0):.0%}")

        # Save results
        result = {
            'success': True,
            'image': image_path,
            'extracted_text': text,
            'medicines': medicines,
            'claude_used': self.claude.available
        }

        output_path = project_root / 'medicines_extracted.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)

        if medicines:
            print("\n✅ Medicines identified:")
            for m in medicines:
                print(f"   • {m.get('medicine_name', 'Unknown')}")
        else:
            print("\n⚠️ No medicines identified")

        print(f"\n💾 Saved: {output_path}")

        return result


# ============ RUN ============

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_test.py <image_path>")
        print("Example: python pipeline_test.py example3.png")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f" Image not found: {image_path}")
        sys.exit(1)

    pipeline = MedicalPipeline(verbose=True)
    result = pipeline.process(image_path)