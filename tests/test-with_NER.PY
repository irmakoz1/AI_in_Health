"""
Simple Medical Prescription Pipeline
Steps:
1. PaddleOCR directly on original image (no preprocessing)
2. Claude API for intelligent medication extraction + similarity matching
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
    """Run PaddleOCR on original image (no preprocessing)"""

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
        """Extract text directly from image (no preprocessing)"""
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


# ============ 2. CLAUDE MEDICAL EXTRACTOR ============

class ClaudeMedicalExtractor:
    """
    Extract medications using Claude API
    """

    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        self.model = model

        # Known medications database for fallback
        self.all_drugs = {
            'warfarin', 'aspirin', 'clopidogrel', 'atorvastatin', 'simvastatin',
            'lisinopril', 'ramipril', 'metoprolol', 'amlodipine', 'furosemide',
            'metformin', 'insulin', 'glipizide', 'pioglitazone', 'sitagliptin',
            'levothyroxine', 'methimazole', 'ibuprofen', 'paracetamol', 'naproxen',
            'diclofenac', 'tramadol', 'morphine', 'amoxicillin', 'azithromycin',
            'ciprofloxacin', 'doxycycline', 'albuterol', 'omeprazole', 'pantoprazole',
            'gabapentin', 'pregabalin', 'carbamazepine', 'sertraline', 'fluoxetine',
            'citalopram', 'escitalopram', 'duloxetine', 'venlafaxine', 'bupropion'
        }

        self.dosage_forms = ['mg', 'mcg', 'μg', 'g', 'ml', 'iu', '%']

        # Claude API setup
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.claude_available = self.api_key is not None

        if not self.claude_available:
            print("   ⚠️ Claude API key not set. Will use similarity matching only.")
        else:
            print(f"   ✅ Claude API ready (model: {self.model})")

    def similarity_score(self, a: str, b: str) -> float:
        """Calculate string similarity"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def find_similar_drug(self, word: str, threshold: float = 0.55) -> Tuple[Optional[str], float]:
        """Find closest matching drug"""
        best_match = None
        best_score = 0

        for drug in self.all_drugs:
            score = self.similarity_score(word, drug)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = drug

        return best_match, best_score

    def extract_with_claude(self, text: str) -> Dict:
        """Use Claude API for intelligent extraction"""

        print("\n" + "=" * 60)
        print("🩺 STEP 2: CLAUDE MEDICAL EXTRACTION")
        print("=" * 60)

        # Clean text
        cleaned = re.sub(r'\s+', ' ', text).strip()
        print(f"   Input text length: {len(cleaned)} chars")

        medications = []
        dosages = []
        claude_result = None

        # Try Claude if available
        if self.claude_available:
            print(f"\n   🤖 Using Claude ({self.model})...")
            claude_result = self._call_claude(cleaned)

            if claude_result and 'medications' in claude_result:
                for med in claude_result['medications']:
                    dosage = med.get('dosage', '')
                    if dosage:
                        value_match = re.findall(r'\d+', dosage)
                        unit_match = re.findall(r'[a-z]+', dosage)
                        dosages.append({
                            'value': value_match[0] if value_match else dosage,
                            'unit': unit_match[0] if unit_match else 'mg',
                            'full_text': dosage
                        })

                    medications.append({
                        'original': med.get('original', med.get('name', '')),
                        'matched': med.get('name', ''),
                        'dosage': dosage,
                        'confidence': med.get('confidence', 0.8),
                        'source': 'claude',
                        'reasoning': med.get('reasoning', '')
                    })

        # Extract dosages with regex (always run)
        dosage_pattern = r'(\d+(?:\.\d+)?)\s*([a-z%]+)?'
        for match in re.finditer(dosage_pattern, cleaned, re.IGNORECASE):
            value = match.group(1)
            unit = match.group(2) if match.group(2) else ''

            is_new = not any(d['full_text'] == match.group(0) for d in dosages)

            if is_new and (unit.lower() in self.dosage_forms or (float(value) <= 500 and float(value) >= 0.5)):
                dosages.append({
                    'value': value,
                    'unit': unit if unit else 'mg',
                    'full_text': match.group(0)
                })

        # Fallback similarity matching if Claude didn't find anything
        if not medications:
            print("\n   📊 Using similarity matching fallback...")
            words = re.findall(r'[A-Za-z][A-Za-z]{3,}', cleaned)
            for word in words:
                if len(word) < 4:
                    continue

                skip = {'the', 'and', 'for', 'with', 'from', 'this', 'that'}
                if word.lower() in skip:
                    continue

                matched, score = self.find_similar_drug(word)
                if matched and score > 0.65:
                    medications.append({
                        'original': word,
                        'matched': matched,
                        'dosage': '',
                        'confidence': score,
                        'source': 'similarity'
                    })

        # Remove duplicates
        unique_meds = []
        seen = set()
        for m in medications:
            key = m['matched']
            if key not in seen:
                seen.add(key)
                unique_meds.append(m)

        # Generate summary
        summary_parts = []
        if unique_meds:
            summary_parts.append(f"Medications: {', '.join([m['matched'] for m in unique_meds])}")
        if dosages:
            summary_parts.append(f"Dosages: {', '.join([d['full_text'] for d in dosages[:3]])}")
        summary = '; '.join(summary_parts) if summary_parts else "No medications identified"

        # Print results
        print(f"\n   💊 Medications found: {len(unique_meds)}")
        for m in unique_meds:
            source_mark = "🤖" if m.get('source') == 'claude' else "📊"
            print(f"      {source_mark} '{m['original']}' → {m['matched']} (conf: {m['confidence']:.0%})")
            if m.get('reasoning'):
                print(f"         💡 {m['reasoning'][:80]}...")

        print(f"\n   💉 Dosages found: {len(dosages)}")
        for d in dosages[:5]:
            print(f"      • {d['full_text']}")

        return {
            'medications': unique_meds,
            'dosages': dosages,
            'summary': summary,
            'claude_used': self.claude_available and claude_result is not None,
            'full_text': cleaned
        }

    def _call_claude(self, text: str) -> Optional[Dict]:
        """Call Claude API"""

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            prompt = f"""You are a medical expert analyzing HANDWRITING OCR output.

OCR Text from prescription: "{text}"

Extract medications and dosages. The text may have OCR errors (e.g., 'Caboigtin' = Carbamazepine).

Available medications: {', '.join(list(self.all_drugs)[:50])}

Return ONLY JSON in this exact format:
{{
  "medications": [
    {{
      "original": "what OCR actually said",
      "name": "correct medication name",
      "dosage": "dosage if found (e.g., '5mg')",
      "confidence": 0.9,
      "reasoning": "brief explanation of match"
    }}
  ]
}}"""

            message = client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.2,
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


# ============ 3. MAIN PIPELINE ============

class MedicalPrescriptionPipeline:
    """Simple pipeline: OCR + Claude extraction"""

    def __init__(self, claude_api_key: str = None, claude_model: str = "claude-3-haiku-20240307", verbose: bool = True):
        self.verbose = verbose
        self.ocr = PaddleOCRRunner(verbose)
        self.extractor = ClaudeMedicalExtractor(claude_api_key, claude_model)

    def process(self, image_path: str) -> Dict:
        """Process prescription image"""

        print("\n" + "=" * 70)
        print("🏥 SIMPLE MEDICAL PRESCRIPTION PIPELINE")
        print(f"   Image: {Path(image_path).name}")
        print("=" * 70)

        # Step 1: OCR directly on original image
        text = self.ocr.extract_text(image_path)

        if not text:
            print("\n❌ No text extracted")
            return {'success': False}

        # Step 2: Claude extraction
        medical_data = self.extractor.extract_with_claude(text)

        # Final results
        result = {
            'success': True,
            'image': image_path,
            'extracted_text': medical_data['full_text'],
            'medications': medical_data['medications'],
            'dosages': medical_data['dosages'],
            'summary': medical_data['summary'],
            'claude_used': medical_data['claude_used']
        }

        # Save results
        with open('prescription_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        print(f"\n{result['summary']}")
        print(f"\n🤖 Claude Used: {'Yes' if result['claude_used'] else 'No'}")
        print("\n💾 Results saved to: prescription_results.json")

        return result


# ============ RUN ============

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_claude_pipeline.py <image_path>")
        print("\nSet your Claude API key:")
        print("  set ANTHROPIC_API_KEY=your-key-here")
        print("\nOr pass as parameter:")
        print("  python simple_claude_pipeline.py image.jpg --api-key your-key")
        sys.exit(1)

    image_path = sys.argv[1]

    # Check for API key
    api_key = None
    if '--api-key' in sys.argv:
        idx = sys.argv.index('--api-key')
        if idx + 1 < len(sys.argv):
            api_key = sys.argv[idx + 1]

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    # Run pipeline
    pipeline = MedicalPrescriptionPipeline(
        claude_api_key=api_key,
        claude_model="claude-3-haiku-20240307",
        verbose=True
    )
    result = pipeline.process(image_path)