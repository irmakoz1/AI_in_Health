"""
Simple Backend for Swiss Medical OCR
Calls your existing pipeline_test.py
"""
import sys
import os
import uuid
import tempfile
import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil

# ============ ADD PARENT DIRECTORY TO PATH ============
# Get the current file's directory
current_dir = Path(__file__).parent
# Add the parent directory to sys.path so Python can find the 'pipeline' module
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

print(f"📁 Current directory: {current_dir}")
print(f"📁 Parent directory added to path: {parent_dir}")
print(f"📁 Contents of parent: {[f.name for f in parent_dir.iterdir() if f.is_dir()]}")

try:
    from pipeline.pipeline_test import MedicalPipeline
    print("✅ Successfully imported MedicalPipeline from pipeline.pipeline_test")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Make sure your pipeline folder is at: {parent_dir / 'pipeline'}")
    sys.exit(1)
app = FastAPI(title="Swiss Medical OCR API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your pipeline
pipeline = MedicalPipeline(verbose=False)  # Set verbose=False to reduce console output

def validate_dosage(drug_name: str, dosage: str, patient_age: int = None) -> dict:
    """Simple dosage validation"""
    findings = []
    drug_lower = drug_name.lower() if drug_name else ""

    # Common warnings
    if "warfarin" in drug_lower:
        findings.append({
            "severity": "WARNING",
            "message": "Warfarin requires regular INR monitoring"
        })

    if "insulin" in drug_lower:
        findings.append({
            "severity": "BLOCK",
            "message": "Insulin dosage requires pharmacist verification"
        })

    if "metformin" in drug_lower and patient_age and patient_age > 80:
        findings.append({
            "severity": "WARNING",
            "message": f"Metformin in elderly (age {patient_age}) - monitor renal function"
        })

    # Check dosage amount
    import re
    numbers = re.findall(r'\d+', dosage)
    if numbers and int(numbers[0]) > 500:
        findings.append({
            "severity": "WARNING",
            "message": f"High dosage {numbers[0]}mg - verify carefully"
        })

    # Determine status
    if any(f["severity"] == "BLOCK" for f in findings):
        status = "blocked"
    elif any(f["severity"] == "WARNING" for f in findings):
        status = "pending_review"
    else:
        status = "ok"

    return {"status": status, "findings": findings}

@app.post("/ocr")
async def process_ocr(
    file: UploadFile = File(...),
    patient_age: str = Form(None),
    patient_weight: str = Form(None),
    allergies: str = Form(""),
    medications: str = Form("")
):
    """
    Process prescription image through your pipeline
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        # Run your pipeline
        result = pipeline.process(temp_path)

        # Extract medicines from pipeline result
        medicines = result.get('medicines', [])
        extracted_text = result.get('extracted_text', '')

        # Format prescriptions for frontend
        prescriptions = []
        validations = []

        for med in medicines:
            prescriptions.append({
                "drug_name": med.get('medicine_name', 'Unknown'),
                "dosage_value": med.get('dosage', ''),
                "dosage_unit": "mg",
                "frequency": "As prescribed",
                "route": "Oral",
                "language": "en",
                "confidence": med.get('confidence', 0.85)
            })

            # Validate dosage
            validation = validate_dosage(
                med.get('medicine_name', ''),
                med.get('dosage', ''),
                int(patient_age) if patient_age else None
            )
            validations.append(validation)

        # Determine overall status
        if any(v['status'] == 'blocked' for v in validations):
            overall_status = "blocked"
        elif any(v['status'] == 'pending_review' for v in validations):
            overall_status = "pending_review"
        else:
            overall_status = "ok" if prescriptions else "pending_review"

        # Create simple FHIR bundle
        fhir_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": []
        }

        for rx in prescriptions:
            fhir_bundle["entry"].append({
                "resource": {
                    "resourceType": "MedicationRequest",
                    "status": "active",
                    "intent": "order",
                    "medicationCodeableConcept": {
                        "text": rx['drug_name']
                    }
                }
            })

        # Return frontend-compatible response
        return JSONResponse({
            "status": overall_status,
            "document_id": str(uuid.uuid4()),
            "prescriptions": prescriptions,
            "validation": validations,
            "fhir_bundle": fhir_bundle,
            "extracted_text": extracted_text[:500] if extracted_text else "",
            "claude_used": True
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Swiss Medical OCR"}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("🚀 Starting Swiss Medical OCR Backend")
    print("=" * 50)
    print(f"📁 Pipeline: pipeline_test.py")
    print(f"🌐 API: http://localhost:8000")
    print(f"📋 Endpoint: POST /ocr")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)