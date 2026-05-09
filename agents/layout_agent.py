"""
Step 1: Layout Analysis for Medical Handwriting
Identifies text regions, prescription areas, and document structure
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add root directory to path
root_dir = Path(__file__).parent.parent if Path(__file__).parent.name == "agents" else Path(__file__).parent
sys.path.insert(0, str(root_dir))

from data_wrapper import MedicalDatasetWrapper


class LayoutAnalyzer:
    """Simple layout analyzer for medical documents"""

    def __init__(self):
        self.results = []

    def analyze_image(self, image_pil):
        """
        Analyze layout of a medical document image

        Args:
            image_pil: PIL Image

        Returns:
            Dictionary with layout information
        """
        # Convert PIL to OpenCV
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        height, width = image_cv.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Apply threshold to find text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours (text regions)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract text regions
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter small regions (noise)
            if area > 500:  # Minimum area for text
                text_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2)
                })

        # Identify document sections based on position
        sections = self._identify_sections(text_regions, height, width)

        # Calculate layout confidence
        confidence = min(1.0, len(text_regions) / 10)  # More regions = higher confidence

        return {
            'image_size': (width, height),
            'num_text_regions': len(text_regions),
            'text_regions': text_regions,
            'sections': sections,
            'confidence': confidence,
            'has_prescription': len(sections.get('prescription', [])) > 0,
            'has_patient_info': len(sections.get('patient_info', [])) > 0
        }

    def _identify_sections(self, regions, height, width):
        """
        Identify different sections of medical document based on position
        """
        sections = {
            'patient_info': [],      # Top of document
            'prescription': [],      # Middle (main content)
            'dosage_instructions': [],  # Lower middle
            'signature': [],         # Bottom
            'unknown': []
        }

        for region in regions:
            x, y, w, h = region['bbox']
            y_position = y / height  # Relative vertical position (0 to 1)

            if y_position < 0.2:  # Top 20%
                sections['patient_info'].append(region)
            elif y_position < 0.6:  # Middle 40%
                sections['prescription'].append(region)
            elif y_position < 0.85:  # Lower middle
                sections['dosage_instructions'].append(region)
            elif y_position >= 0.85:  # Bottom 15%
                sections['signature'].append(region)
            else:
                sections['unknown'].append(region)

        return sections

    def draw_layout(self, image_pil, layout_result, save_path=None):
        """
        Draw bounding boxes on image to visualize layout
        """
        # Convert PIL to RGB for drawing
        img_draw = image_pil.copy()

        # Colors for different sections
        colors = {
            'patient_info': (255, 0, 0),    # Blue
            'prescription': (0, 255, 0),    # Green
            'dosage_instructions': (0, 255, 255),  # Yellow
            'signature': (255, 0, 255),     # Purple
            'unknown': (128, 128, 128)      # Gray
        }

        # Draw all text regions
        for section_name, regions in layout_result['sections'].items():
            color = colors.get(section_name, (0, 0, 255))

            for region in regions:
                x, y, w, h = region['bbox']

                # Draw rectangle
                draw_rect = ImageDraw.Draw(img_draw)
                draw_rect.rectangle([x, y, x+w, y+h], outline=color, width=2)

                # Add label
                draw_rect.text((x, y-10), section_name, fill=color)

        if save_path:
            img_draw.save(save_path)

        return img_draw


# Simple drawing function (since we can't use ImageDraw easily)
def draw_boxes_cv(image_pil, layout_result):
    """Draw bounding boxes using OpenCV"""
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    colors = {
        'patient_info': (255, 0, 0),
        'prescription': (0, 255, 0),
        'dosage_instructions': (0, 255, 255),
        'signature': (255, 0, 255),
        'unknown': (128, 128, 128)
    }

    for section_name, regions in layout_result['sections'].items():
        color = colors.get(section_name, (0, 0, 255))

        for region in regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(image_cv, (x, y), (x+w, y+h), color, 2)

            # Add label
            cv2.putText(image_cv, section_name, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image_cv


def main():
    """Main pipeline step 1"""
    print("=" * 60)
    print("🏥 STEP 1: LAYOUT ANALYSIS")
    print("=" * 60)

    # Load dataset
    print("\n📚 Loading dataset...")
    dataset = MedicalDatasetWrapper()

    # Initialize layout analyzer
    analyzer = LayoutAnalyzer()

    # Create output directory
    output_dir = Path("layout_analysis_results")
    output_dir.mkdir(exist_ok=True)

    # Process first N records
    num_records = min(10, len(dataset))
    print(f"\n🔍 Analyzing layout for {num_records} records...")

    results_summary = []

    for i in range(num_records):
        print(f"\n📄 Processing record {i+1}/{num_records}")

        # Get record
        record = dataset[i]
        print(f"   ID: {record.record_id}")

        try:
            # Get image
            image = record.get_image_as_pil()
            print(f"   Image size: {image.size}")

            # Analyze layout
            layout = analyzer.analyze_image(image)

            # Print results
            print(f"   📊 Layout analysis:")
            print(f"      - Text regions found: {layout['num_text_regions']}")
            print(f"      - Has prescription area: {layout['has_prescription']}")
            print(f"      - Has patient info: {layout['has_patient_info']}")
            print(f"      - Confidence: {layout['confidence']:.2%}")

            # Show section breakdown
            for section, regions in layout['sections'].items():
                if regions:
                    print(f"      - {section}: {len(regions)} regions")

            # Visualize and save
            vis_image = draw_boxes_cv(image, layout)
            output_path = output_dir / f"{record.record_id}_layout.jpg"
            cv2.imwrite(str(output_path), vis_image)
            print(f"      💾 Saved visualization: {output_path}")

            # Save results
            results_summary.append({
                'record_id': record.record_id,
                'num_regions': layout['num_text_regions'],
                'has_prescription': layout['has_prescription'],
                'has_patient_info': layout['has_patient_info'],
                'confidence': layout['confidence'],
                'sections': {k: len(v) for k, v in layout['sections'].items()}
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("📊 LAYOUT ANALYSIS SUMMARY")
    print("=" * 60)

    if results_summary:
        avg_regions = np.mean([r['num_regions'] for r in results_summary])
        avg_confidence = np.mean([r['confidence'] for r in results_summary])
        has_prescription_pct = sum([r['has_prescription'] for r in results_summary]) / len(results_summary) * 100

        print(f"\n✅ Processed: {len(results_summary)} records")
        print(f"📐 Average text regions: {avg_regions:.1f}")
        print(f"🎯 Average confidence: {avg_confidence:.2%}")
        print(f"💊 Records with prescription area: {has_prescription_pct:.1f}%")

        # Section statistics
        print(f"\n📋 Section detection rates:")
        section_counts = {}
        for r in results_summary:
            for section, count in r['sections'].items():
                if count > 0:
                    section_counts[section] = section_counts.get(section, 0) + 1

        for section, count in section_counts.items():
            rate = count / len(results_summary) * 100
            print(f"   {section}: {rate:.1f}%")

    print("\n" + "=" * 60)
    print(" STEP 1 COMPLETE!")
    print(f" Results saved in: {output_dir}")
    print("=" * 60)

    return results_summary


if __name__ == "__main__":
    # Need this for drawing
    from PIL import ImageDraw

    results = main()

    print("\n Next step: Run STEP 2 - Handwriting Recognition (OCR)")