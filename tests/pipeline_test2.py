"""
Medical Prescription Pipeline - Auto-detects Claude model
"""

import re
import json
import sys
from pathlib import Path
import os
from typing import List, Dict, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    if Path('.env').exists():
        load_dotenv()
        print("✅ Loaded API key from .env file")
except:
    pass

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
            print("   ❌ PaddleOCR not installed")
            self.available = False

    def extract_text_with_confidence(self, image_path: str) -> Dict:
        if not self.available:
            return {"full_text": "", "lines": [], "avg_confidence": 0.0}

        if self.verbose:
            print("\n" + "=" * 60)
            print("📝 STEP 1: PADDLEOCR EXTRACTION")
            print("=" * 60)

        try:
            result = self.ocr.ocr(image_path, cls=True)

            if result and result[0]:
                lines = []
                confidences = []

                for line in result[0]:
                    if line and len(line) >= 2:
                        if isinstance(line[1], tuple):
                            text = line[1][0]
                            confidence = line[1][1]
                        else:
                            text = str(line[1])
                            confidence = 0.5

                        lines.append({'text': text, 'confidence': confidence})
                        confidences.append(confidence)

                full_text = '\n'.join([l['text'] for l in lines])
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                if self.verbose:
                    print(f"   ✅ Extracted {len(full_text)} characters")
                    print(f"   📊 Avg confidence: {avg_confidence:.2%}")
                    print(f"   Preview: {full_text[:200]}...")

                return {
                    'full_text': full_text,
                    'lines': lines,
                    'avg_confidence': avg_confidence,
                    'success': True
                }
            else:
                print("   ⚠️ No text detected")
                return {"full_text": "", "lines": [], "avg_confidence": 0.0, "success": False}

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {"full_text": "", "lines": [], "avg_confidence": 0.0, "success": False}


# ============ CLAUDE EXTRACTOR WITH AUTO MODEL ============

class ClaudeMedicineExtractor:
    def __init__(self, api_key: str = None):
        # List of possible model names to try (in order)
        self.models_to_try = [
            "claude-3-haiku-20240307",      # Original Haiku
            "claude-3-sonnet-20240229",     # Original Sonnet
            "claude-3-opus-20240229",       # Original Opus
            "claude-3-5-haiku-20241022",    # New Haiku
            "claude-3-5-sonnet-20241022",   # New Sonnet
        ]

        self.active_model = None
        self.fuzzy_matches = {
            'stalevo': ['stalev', 'stal evo', 'stalevo', 'stalev0', 'STALEV'],
            'levodopa': ['levodopa', 'levo dopa', 'l-dopa', 'levodop'],
            'carbidopa': ['carbidopa', 'calbidopa', 'carbidop'],
        }

        # Load API key
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.claude_available = self.api_key is not None

        if not self.claude_available:
            print("   ⚠️ Claude API key not found. Using fuzzy matching only.")
        else:
            key_preview = f"...{self.api_key[-8:]}" if len(self.api_key) > 8 else "***"
            print(f"   ✅ Claude API key loaded (ending: {key_preview})")
            # Try to find a working model
            self._find_working_model()

    def _find_working_model(self):
        """Try different models to find one that works"""
        try:
            import anthropic

            for model_name in self.models_to_try:
                try:
                    # Quick test with a simple prompt
                    client = anthropic.Anthropic(api_key=self.api_key)
                    test_response = client.messages.create(
                        model=model_name,
                        max_tokens=10,
                        messages=[{"role": "user", "content": "Test"}]
                    )
                    self.active_model = model_name
                    print(f"   ✅ Claude API ready (model: {self.active_model})")
                    return
                except Exception as e:
                    continue

            # If no model works
            print(f"   ⚠️ No working Claude model found. Using fuzzy matching only.")
            self.claude_available = False

        except ImportError:
            print("   ❌ anthropic not installed")
            self.claude_available = False

    def extract_medicines_with_confidence(self, text: str, ocr_lines: List[Dict] = None) -> Dict:
        print("\n" + "=" * 60)
        print("🩺 STEP 2: MEDICINE EXTRACTION")
        print("=" * 60)

        cleaned = re.sub(r'\s+', ' ', text).strip()
        print(f"   Input: {cleaned[:200]}...")

        medicines = []

        # Fuzzy matching
        print(f"\n   📊 Running fuzzy matching...")
        fuzzy_matches = self._fuzzy_match_medicines(cleaned)
        for match in fuzzy_matches:
            ocr_conf = self._get_ocr_confidence(match['ocr_text'], ocr_lines) if ocr_lines else 0.7
            match['confidence'] = (match['confidence'] + ocr_conf) / 2
            medicines.append(match)

        # Claude API (if available and we have a working model)
        if self.claude_available and self.active_model:
            print(f"\n   🤖 Claude analyzing with model: {self.active_model}")
            claude_result = self._call_claude(cleaned)

            if claude_result and claude_result.get('medicines'):
                for med in claude_result['medicines']:
                    ocr_conf = self._get_ocr_confidence(med.get('ocr_text', ''), ocr_lines) if ocr_lines else 0.7
                    claude_conf = med.get('confidence', 0.7)

                    medicines.append({
                        'ocr_text': med.get('ocr_text', ''),
                        'medicine_name': med.get('medicine_name', ''),
                        'dosage': med.get('dosage', ''),
                        'confidence': (claude_conf + ocr_conf) / 2,
                        'reasoning': med.get('reasoning', ''),
                        'source': 'claude'
                    })

        # Deduplicate
        unique_meds = self._deduplicate(medicines)
        dosages = self._extract_dosages(cleaned)
        overall_conf = self._calculate_overall_confidence(unique_meds, cleaned)

        # Print results
        print(f"\n   💊 Medicines found:")
        if unique_meds:
            for m in unique_meds:
                print(f"\n      • '{m['ocr_text']}' → {m['medicine_name']}")
                if m.get('dosage'):
                    print(f"        Dosage: {m['dosage']}")
                print(f"        Confidence: {m['confidence']:.0%}")
        else:
            print("      None found")

        if dosages:
            print(f"\n   💉 Dosages: {', '.join(dosages)}")

        print(f"\n   📊 Overall confidence: {overall_conf:.0%}")

        return {
            'medicines': unique_meds,
            'dosages': dosages,
            'overall_confidence': overall_conf,
            'claude_used': self.claude_available and self.active_model is not None
        }

    def _fuzzy_match_medicines(self, text: str) -> List[Dict]:
        medicines = []
        text_lower = text.lower()

        for correct_name, variants in self.fuzzy_matches.items():
            for variant in variants:
                if variant in text_lower:
                    match = re.search(re.escape(variant), text, re.IGNORECASE)
                    ocr_text = match.group(0) if match else variant

                    medicines.append({
                        'ocr_text': ocr_text,
                        'medicine_name': correct_name.upper(),
                        'dosage': self._find_dosage(text, match.start() if match else 0),
                        'confidence': 0.85,
                        'source': 'fuzzy'
                    })
                    break

        return medicines

    def _get_ocr_confidence(self, search_text: str, ocr_lines: List[Dict]) -> float:
        if not ocr_lines:
            return 0.7
        for line in ocr_lines:
            if search_text.lower() in line['text'].lower():
                return line['confidence']
        return 0.5

    def _find_dosage(self, text: str, position: int) -> str:
        if position == -1:
            return ""
        start = max(0, position - 20)
        end = min(len(text), position + 40)
        surrounding = text[start:end]

        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml)?', surrounding, re.IGNORECASE)
        if match:
            return match.group(0)

        slash_match = re.search(r'(/\d+)+', surrounding)
        if slash_match:
            return slash_match.group(0)

        return ""

    def _call_claude(self, text: str) -> Optional[Dict]:
        """Call Claude API with working model"""
        if not self.active_model:
            return None

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            prompt = f"""You are analyzing HANDWRITTEN medical prescription OCR text.

OCR Text: "{text}"

Find medicines. OCR may have errors (e.g., "STALEV0125" = STALEVO).

Return ONLY JSON:
{{
  "medicines": [
    {{
      "ocr_text": "exact OCR text",
      "medicine_name": "correct name",
      "dosage": "dosage if found",
      "confidence": 0.9,
      "reasoning": "explanation"
    }}
  ]
}}"""

            message = client.messages.create(
                model=self.active_model,
                max_tokens=1000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return None

        except Exception as e:
            print(f"   ⚠️ Claude error with {self.active_model}: {e}")
            return None

    def _deduplicate(self, medicines: List[Dict]) -> List[Dict]:
        seen = {}
        for m in medicines:
            name = m['medicine_name']
            if name not in seen or m['confidence'] > seen[name]['confidence']:
                seen[name] = m
        return list(seen.values())

    def _calculate_overall_confidence(self, medicines: List[Dict], text: str) -> float:
        if not medicines:
            return 0.0
        avg_conf = sum(m['confidence'] for m in medicines) / len(medicines)
        return min(1.0, avg_conf)

    def _extract_dosages(self, text: str) -> List[str]:
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml)',
            r'/(\d+)(?:/|$)',
        ]
        dosages = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    dosages.append(f"{match[0]}{match[1] if len(match) > 1 else 'mg'}")
                else:
                    dosages.append(f"{match}mg")
        return list(set(dosages[:5]))


# ============ MAIN PIPELINE ============

class MedicalPrescriptionPipeline:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.ocr = PaddleOCRRunner(verbose)
        self.extractor = ClaudeMedicineExtractor()

    def process(self, image_path: str) -> Dict:
        print("\n" + "=" * 70)
        print("🏥 MEDICAL PRESCRIPTION PIPELINE")
        print(f"   Image: {Path(image_path).name}")
        print("=" * 70)

        ocr_result = self.ocr.extract_text_with_confidence(image_path)

        if not ocr_result['success']:
            return {'success': False}

        medical_data = self.extractor.extract_medicines_with_confidence(
            ocr_result['full_text'],
            ocr_result['lines']
        )

        result = {
            'success': True,
            'image': image_path,
            'ocr_confidence': ocr_result['avg_confidence'],
            'medicines': medical_data['medicines'],
            'dosages': medical_data['dosages'],
            'overall_confidence': medical_data['overall_confidence']
        }

        with open('medicines_extracted.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        print(f"\n🎯 OCR Confidence: {result['ocr_confidence']:.0%}")
        print(f"🎯 Overall Confidence: {result['overall_confidence']:.0%}")

        if result['medicines']:
            print("\n✅ Medicines found:")
            for m in result['medicines']:
                print(f"   • {m['medicine_name']} (conf: {m['confidence']:.0%})")
        else:
            print("\n⚠️ No medicines found")

        return result


# ============ RUN ============

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_auto_model.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    pipeline = MedicalPrescriptionPipeline(verbose=True)
    result = pipeline.process(image_path)