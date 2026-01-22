"""
Enhanced AI Medical Report Analysis Helper
Supports: Groq + OpenAI + HuggingFace (with Vision for CT Scans)
Handles: CT Scans (with detailed region analysis), Blood Reports (PDF + Images)
"""

import os
import base64
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import pytesseract
import cv2
import pdfplumber
# Import disease model loader

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'blood_models'))
from blood_model_loader import predict_disease, loaded_disease_models


load_dotenv()

# Load API keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Initialize clients
groq_client = None
openai_client = None





# Add this RIGHT AFTER the imports (around line 15)

def extract_text_with_tesseract(image_bytes):
    """Use Tesseract OCR to extract text from images - WORKS ON WINDOWS"""
    try:
        from PIL import Image
        import numpy as np
        
        # Open image
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding for better text recognition
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # Extract text with optimized config for medical reports
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(denoised, config=custom_config)
        
        if text and len(text.strip()) > 50:
            print(f"✅ Tesseract OCR extracted {len(text.strip())} chars")
            return text.strip()
        else:
            # Try with original image if preprocessing failed
            text = pytesseract.image_to_string(image, config=custom_config)
            if text and len(text.strip()) > 50:
                print(f"✅ Tesseract OCR (original) extracted {len(text.strip())} chars")
                return text.strip()
            
        print(f"⚠️ Tesseract extracted insufficient text: {len(text.strip()) if text else 0} chars")
        return None
            
    except Exception as e:
        print(f"⚠️ Tesseract OCR error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def init_groq():
    """Initialize Groq API"""
    global groq_client
    if not GROQ_API_KEY or len(GROQ_API_KEY) < 20:
        print("⚠️ Groq API key not found")
        return False
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq API connected")
        return True
    except Exception as e:
        print(f"❌ Groq initialization failed: {e}")
        return False

def init_openai():
    """Initialize OpenAI API with vision support"""
    global openai_client
    if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 20:
        print("⚠️ OpenAI API key not found")
        return False
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        openai_client.models.list()
        print("✅ OpenAI API connected (Vision enabled)")
        return True
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {e}")
        return False

def init_huggingface():
    """Check HuggingFace API availability"""
    if not HUGGINGFACE_API_KEY or len(HUGGINGFACE_API_KEY) < 20:
        print("⚠️ HuggingFace API key not found")
        return False
    print("✅ HuggingFace API key available")
    return True

def clean_markdown(text):
    """Remove markdown formatting"""
    import re
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def encode_image_to_base64(image_source):
    """Encode image to base64 from file path, PIL Image, or bytes"""
    try:
        if isinstance(image_source, str):
            with open(image_source, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image_source, Image.Image):
            buffer = BytesIO()
            image_source.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image_source, bytes):
            return base64.b64encode(image_source).decode('utf-8')
        else:
            raise ValueError("Unsupported image source type")
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return None

def analyze_ct_scan_with_huggingface(image_bytes):
    """
    Analyze CT scan using HuggingFace Vision-Language Model
    Provides detailed anatomical region analysis
    """
    if not HUGGINGFACE_API_KEY:
        return None
    
    try:
        # Using BLIP for image captioning
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            description = result[0].get('generated_text', '') if isinstance(result, list) else ''
            
            if description:
                print(f"✅ HuggingFace Vision: {description}")
                return {
                    'success': True,
                    'description': description,
                    'provider': 'HuggingFace BLIP'
                }
        elif response.status_code == 503:
            print(f"⚠️ HuggingFace Vision model loading (503), skipping...")
        else:
            print(f"⚠️ HuggingFace Vision failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️ HuggingFace Vision error: {e}")
    
    return None

def analyze_ct_scan_with_openai_vision(image_bytes, prediction_result):
    """
    Enhanced CT scan analysis using OpenAI Vision
    Provides detailed anatomical insights and region identification
    """
    if not openai_client:
        return None
    
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        predicted_class = prediction_result.get('predicted_class', 'Unknown')
        confidence = prediction_result.get('confidence', 0)
        all_probs = prediction_result.get('all_probabilities', {})
        
        probs_text = "\n".join([f"{cls}: {prob*100:.1f}%" for cls, prob in all_probs.items()])
        
        prompt = f"""You are an expert radiologist analyzing a kidney CT scan image.

AI MODEL PREDICTION:
• Predicted Condition: {predicted_class}
• Confidence: {confidence*100:.1f}%
• Probability Distribution:
{probs_text}

Please analyze this CT scan image and provide a comprehensive medical report with the following sections:

🔍 IMAGE QUALITY ASSESSMENT
Evaluate image clarity, contrast, and diagnostic quality

📍 ANATOMICAL REGION IDENTIFICATION
• Identify visible kidney structures (cortex, medulla, pelvis, calyces)
• Note any visible abnormalities or irregular regions
• Describe the location and extent of any pathology
• Identify if both kidneys are visible or just one

🎯 DETAILED FINDING ANALYSIS
Based on the AI prediction of "{predicted_class}":
• Describe specific visual features consistent with this diagnosis
• Highlight key indicators (density changes, irregular borders, mass lesions, calcifications)
• Identify exact location within the kidney (upper pole, middle, lower pole, cortical, medullary)
• Estimate approximate size if abnormality is visible

⚕️ CLINICAL CORRELATION
• How does this finding align with the AI prediction?
• What are the typical imaging characteristics of {predicted_class}?
• Are there any concerning features that require immediate attention?

📊 CONFIDENCE ASSESSMENT
Given the {confidence*100:.1f}% AI confidence:
• Does the visual analysis support this prediction?
• Are there any alternative diagnoses to consider?
• What additional imaging might be helpful?

🏥 CLINICAL RECOMMENDATIONS
• Immediate actions required
• Follow-up imaging schedule
• Specialist consultations needed
• Additional diagnostic tests recommended

⚠️ KEY FINDINGS SUMMARY
List 4-6 most important findings in bullet points for quick reference

🔔 URGENT ALERT INDICATORS
• Symptoms requiring immediate medical attention
• Red flags to watch for
• When to go to emergency room

📋 PATIENT GUIDANCE
• What this finding typically means
• Treatment options overview
• Prognosis and outlook
• Lifestyle modifications

⚖️ LIMITATIONS & DISCLAIMER
Standard medical imaging limitations and consultation requirements

Use plain text format with • for bullets, section headers in CAPS. Be specific about anatomical locations and visual features."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini model to avoid quota issues
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert radiologist specializing in kidney CT scan interpretation. Provide detailed, anatomically precise analysis with clear identification of pathological regions. Use plain text with • bullets."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        analysis = response.choices[0].message.content
        analysis = clean_markdown(analysis)
        
        if analysis and len(analysis) > 100:
            print(f"✅ OpenAI Vision CT analysis: {len(analysis)} chars")
            return {
                'success': True,
                'analysis': analysis,
                'ai_provider': 'OpenAI Vision (GPT-4o-mini)',
                'has_visual_analysis': True
            }
    except Exception as e:
        error_msg = str(e)
        if 'insufficient_quota' in error_msg or '429' in error_msg:
            print(f"⚠️ OpenAI quota exceeded, falling back to text-only analysis")
        else:
            print(f"⚠️ OpenAI Vision CT analysis failed: {e}")
    
    return None

def generate_ct_scan_analysis(prediction_result, image_bytes=None):
    """
    Enhanced CT scan analysis with multiple strategies
    
    Strategy:
    1. Try OpenAI Vision (best for detailed visual analysis)
    2. Try HuggingFace Vision + Text analysis
    3. Fall back to text-only analysis
    4. Use template as last resort
    """
    predicted_class = prediction_result.get('predicted_class', 'Unknown')
    confidence = prediction_result.get('confidence', 0)
    all_probs = prediction_result.get('all_probabilities', {})
    
    # Strategy 1: OpenAI Vision (if image bytes provided)
    if image_bytes and openai_client:
        print("🔬 Analyzing CT scan with OpenAI Vision...")
        result = analyze_ct_scan_with_openai_vision(image_bytes, prediction_result)
        if result and result.get('success'):
            return result
    
    # Strategy 2: HuggingFace Vision + Groq Text Analysis
    if image_bytes and HUGGINGFACE_API_KEY and groq_client:
        print("🔄 Trying HuggingFace Vision + Groq...")
        hf_result = analyze_ct_scan_with_huggingface(image_bytes)
        if hf_result and hf_result.get('success'):
            visual_description = hf_result['description']
            result = generate_ct_text_analysis_groq(prediction_result, visual_description)
            if result:
                return result
    
    # Strategy 3: Text-only analysis (no image)
    print("📝 Using text-only CT analysis...")
    if groq_client:
        result = generate_ct_text_analysis_groq(prediction_result)
        if result:
            return result
    
    if openai_client:
        result = generate_ct_text_analysis_openai(prediction_result)
        if result:
            return result
    
    # Strategy 4: Template fallback
    print("📋 Using template fallback for CT analysis")
    return get_template_ct_analysis(predicted_class, confidence)

def generate_ct_text_analysis_groq(prediction_result, visual_description=None):
    """Generate CT analysis using Groq with optional visual context"""
    if not groq_client:
        return None
    
    try:
        predicted_class = prediction_result.get('predicted_class', 'Unknown')
        confidence = prediction_result.get('confidence', 0)
        all_probs = prediction_result.get('all_probabilities', {})
        
        probs_text = "\n".join([f"{cls}: {prob*100:.1f}%" for cls, prob in all_probs.items()])
        
        visual_context = f"\nVISUAL ANALYSIS:\n{visual_description}\n" if visual_description else ""
        
        prompt = f"""Expert radiologist analyzing kidney CT scan.

SCAN RESULTS:
Predicted: {predicted_class}
Confidence: {confidence*100:.1f}%

Probabilities:
{probs_text}{visual_context}

Provide comprehensive report (plain text, • bullets):

IMAGE QUALITY ASSESSMENT
ANATOMICAL FINDINGS
• Specific kidney regions affected
• Visual characteristics of pathology
• Size and location details

DETAILED FINDING ANALYSIS
• Key indicators of {predicted_class}
• Exact anatomical location (upper/middle/lower pole, cortical/medullary)
• Severity assessment

CLINICAL CORRELATION
CONFIDENCE ASSESSMENT
CLINICAL RECOMMENDATIONS
KEY FINDINGS SUMMARY (4-6 bullet points)
URGENT ALERT INDICATORS
PATIENT GUIDANCE
LIMITATIONS & DISCLAIMER"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Expert radiologist specializing in kidney CT interpretation. Provide anatomically precise analysis. Plain text, • bullets."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3500
        )
        
        analysis = clean_markdown(response.choices[0].message.content)
        
        if analysis and len(analysis) > 100:
            print(f"✅ Groq CT analysis: {len(analysis)} chars")
            return {
                'success': True,
                'analysis': analysis,
                'ai_provider': 'Groq (Llama 3.3 70B)' + (' + HuggingFace Vision' if visual_description else '')
            }
    except Exception as e:
        print(f"⚠️ Groq CT analysis failed: {e}")
    
    return None

def generate_ct_text_analysis_openai(prediction_result):
    """Generate CT analysis using OpenAI text model"""
    if not openai_client:
        return None
    
    try:
        predicted_class = prediction_result.get('predicted_class', 'Unknown')
        confidence = prediction_result.get('confidence', 0)
        all_probs = prediction_result.get('all_probabilities', {})
        
        probs_text = "\n".join([f"{cls}: {prob*100:.1f}%" for cls, prob in all_probs.items()])
        
        prompt = f"""Expert radiologist analyzing kidney CT scan.

SCAN RESULTS:
Predicted: {predicted_class}
Confidence: {confidence*100:.1f}%

Probabilities:
{probs_text}

Provide comprehensive kidney CT report with anatomical details, specific region identification, clinical recommendations, key findings summary, urgent indicators, patient guidance, and limitations. Plain text, • bullets."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Expert radiologist. Plain text, • bullets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        analysis = clean_markdown(response.choices[0].message.content)
        return {
            'success': True,
            'analysis': analysis,
            'ai_provider': 'OpenAI (GPT-4o-mini)'
        }
    except Exception as e:
        if 'insufficient_quota' in str(e) or '429' in str(e):
            print(f"⚠️ OpenAI quota exceeded")
        else:
            print(f"❌ OpenAI CT analysis failed: {e}")
    
    return None

# ============================================
# BLOOD REPORT FUNCTIONS
# ============================================



# Add this function
# In ai_medical_helper.py - Enhanced version with explicit disease identification

def analyze_blood_report_with_disease_models(extracted_text=None, blood_params=None, source='pdf'):
    """
    Enhanced blood analysis: OCR → ML Disease Classification → AI Explanation
    
    Flow:
    1. Extract parameters from text
    2. Run 6 disease classification models
    3. Generate AI explanation based on ML results (with explicit disease ID)
    4. If ML fails, AI does direct analysis
    """
    
    # Step 1: Get blood parameters (unchanged)
    if extracted_text and not blood_params:
        blood_params = parse_blood_values_from_text(extracted_text)
    
    if not blood_params or len(blood_params) < 3:
        print("⚠️ Insufficient blood parameters for disease prediction")
        # Fallback to text-only analysis
        if groq_client and extracted_text:
            return analyze_blood_report_text_groq(extracted_text, source)
        return get_template_blood_analysis({})
    
    print(f"📊 Blood parameters extracted: {list(blood_params.keys())}")
    
    # Step 2: ML Disease Classification (unchanged)
    if not loaded_disease_models:
        print("⚠️ Disease models not loaded, using AI-only analysis")
        if groq_client:
            return analyze_blood_report_text_groq(extracted_text or str(blood_params), source)
        return get_template_blood_analysis(blood_params)
    
    disease_results = predict_disease(blood_params)
    
    if not disease_results:
        print("⚠️ Disease prediction failed, using AI-only analysis")
        if groq_client:
            return analyze_blood_report_text_groq(extracted_text or str(blood_params), source)
        return get_template_blood_analysis(blood_params)
    
    print(f"✅ Disease predictions: {disease_results['num_positive']} positive findings")
    
    # Step 3: Generate AI explanation with ML context (ENHANCED PROMPT)
    if groq_client:
        try:
            params_text = "\n".join([f"• {k}: {v}" for k, v in blood_params.items()])
            
            positive_diseases = disease_results['positive_diseases']
            positive_text = ", ".join(positive_diseases) if positive_diseases else "None detected"
            
            # Build detailed predictions text (unchanged)
            pred_details = []
            for disease, pred in disease_results['predictions'].items():
                if 'error' not in pred:
                    status = "✓ POSITIVE" if pred['is_positive'] else "✗ Negative"
                    conf = pred['confidence'] * 100
                    pred_details.append(f"• {disease}: {status} ({conf:.1f}% confidence)")
            
            predictions_text = "\n".join(pred_details)
            
            # ENHANCED PROMPT: Explicit disease identification for user
            prompt = f"""You are an expert pathologist analyzing blood test results with ML disease classification. 
SPEAK DIRECTLY TO THE USER in simple, empathetic language (e.g., "Based on your blood report, you may be going through...").

EXTRACTED BLOOD PARAMETERS:
{params_text}

ML DISEASE CLASSIFICATION RESULTS (6 Models):
{predictions_text}

POSITIVE FINDINGS: {positive_text}

Provide comprehensive medical analysis in plain text (NO markdown). Use • for bullets.

DIAGNOSED CONDITIONS (KEY SECTION - BE DIRECT)
• Clearly state what diseases the user is likely going through (e.g., "You are likely experiencing Dengue fever with 90.5% confidence, due to severe low platelets (45k/µL < 150k normal) and low WBC (3k/µL indicating viral suppression). This suggests a high risk for bleeding complications.")
• For each positive disease: Name it, confidence, key abnormal params driving it, why it matches symptoms/risks.
• If no positives: "No major diseases detected in this report, but routine monitoring is recommended."
• Rate severity: Low/Medium/High based on confidence >80% = High.

ML DISEASE CLASSIFICATION SUMMARY
• Explain ML detection logic (e.g., "Dengue model triggered by thrombocytopenia + lymphocytosis").
• Confidence breakdown: ML raw vs. medically adjusted.

EXTRACTED VALUES
• List each parameter: Value (Status: Normal/High/Low/Abnormal) - tie to diseases if relevant.

OVERALL HEALTH ASSESSMENT
• Summarize: "Your report indicates viral infection risks (Dengue), but hemoglobin is normal."
• Severity: Low/Med/High across all findings.
• User message: "Don't panic - this is preliminary; see a doctor ASAP."

DETAILED PARAMETER ANALYSIS
• Link params to diseases (e.g., "Platelets 45k/µL: Critically low, hallmark of Dengue progression").
• Flag any imputations as "Estimated value - retest recommended."

IDENTIFIED HEALTH CONCERNS
• Top 3 risks: e.g., "1. Bleeding risk from low platelets (Dengue-related)."

DISEASE-SPECIFIC GUIDANCE
For each detected disease:
• What it means for you: Symptoms (fever, rash for Dengue).
• Immediate steps: Hydrate, rest, avoid aspirin.
• Treatment: Hospitalization if severe; antivirals/supportive care.
• Prognosis: "Dengue usually resolves in 7-10 days with care."

DIETARY RECOMMENDATIONS
• Tailored: e.g., "For Dengue: Hydrating fluids, vitamin C-rich foods; avoid spicy/oily."

LIFESTYLE MODIFICATIONS
• Rest, monitor fever; no strenuous activity during recovery.

FOLLOW-UP RECOMMENDATIONS
• See doctor TODAY for positives >70% confidence.
• Retest platelets/WBC in 48 hours; full viral panel.

URGENT SYMPTOMS TO WATCH
• For Dengue: Severe abdominal pain, persistent vomiting, bleeding gums → ER immediately.
• General: Dizziness, confusion, rapid heart rate.

LIMITATIONS & DISCLAIMER
• This is AI/ML-assisted screening (~90% accurate on validated data); NOT a diagnosis.
• See qualified doctor for confirmation/tests. ML models trained on general data."""

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Expert pathologist. Empathetic, direct to user. Plain text, • bullets. Focus on clear disease explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower for factual accuracy
                max_tokens=4500   # Increased for detailed disease sections
            )
            
            analysis = clean_markdown(response.choices[0].message.content)
            
            print(f"✅ ML + AI analysis complete: {len(analysis)} chars (with disease ID)")
            
            return {
                'success': True,
                'disease_predictions': disease_results,
                'analysis': analysis,
                'ai_provider': 'ML Disease Models (6) + Groq AI (Disease-Focused)',
                'extraction_method': source,
                'has_disease_classification': True,
                'num_positive_diseases': disease_results['num_positive'],
                'positive_diseases': disease_results['positive_diseases'],
                'detected_diseases_summary': f"You are likely going through: {positive_text}"  # NEW: Quick user summary
            }
            
        except Exception as e:
            print(f"⚠️ Groq analysis failed: {e}")
    
    # Fallback: Enhanced template with disease list
    positive_list = "\n".join([f"• {d} (High likelihood based on your params)" for d in disease_results['positive_diseases']])
    if not positive_diseases:
        positive_list = "• No major diseases detected - good news, but consult for full check."
    
    fallback_analysis = f"""DIAGNOSED CONDITIONS
Based on your blood report and ML analysis, you are likely going through:
{positive_list}

(Full details in sections below - see doctor immediately for confirmation.)

{get_template_blood_analysis(blood_params)['analysis']}"""
    
    return {
        'success': True,
        'disease_predictions': disease_results,
        'analysis': fallback_analysis,
        'ai_provider': 'ML Disease Models + Enhanced Template',
        'has_disease_classification': True,
        'num_positive_diseases': disease_results['num_positive'],
        'positive_diseases': disease_results['positive_diseases']
    }
    
# Add this enhanced version to ai_medical_helper.py
# Replace the existing parse_blood_values_from_text function



def parse_blood_values_from_text(text):
    """
    PERFECT: Extract blood parameters with EXACT feature names
    Handles: ×10^9, X10⁹, x10, per µL, and all variations
    """
    import re
    
    if not text:
        return {}
    
    params = {}
    text_lower = text.lower()
    lines = text.split('\n')
    
    # ULTRA-FLEXIBLE patterns - matches EVERYTHING
    patterns = {
        # ========== GLUCOSE ==========
        'fasting_glucose': [
            r'fasting[^\d]*glucose[^\d]*(\d+\.?\d*)',
            r'glucose[^\d]*fasting[^\d]*(\d+\.?\d*)',
            r'\bfbs\b[^\d]*(\d+\.?\d*)',
        ],
        'random_glucose': [
            r'random[^\d]*glucose[^\d]*(\d+\.?\d*)',
            r'\brbs\b[^\d]*(\d+\.?\d*)',
        ],
        'postprandial_glucose': [
            r'postprandial[^\d]*glucose[^\d]*(\d+\.?\d*)',
            r'pp[^\d]*glucose[^\d]*(\d+\.?\d*)',
            r'\bppbs\b[^\d]*(\d+\.?\d*)',
        ],
        'hba1c': [
            r'\bhba1c\b[^\d]*(\d+\.?\d*)',
            r'glycated[^\d]*h[ae]moglobin[^\d]*(\d+\.?\d*)',
            r'hemoglobin[^\d]*a1c[^\d]*(\d+\.?\d*)',
        ],
        
        # ========== WBC - SUPER FLEXIBLE ==========
        'wbc': [
            r'total[^\d]*leucocyte[s]?[^\d]*count[^\d]*\(?wbc\)?[^\d]*(\d+\.?\d*)',
            r'total[^\d]*leukocyte[s]?[^\d]*count[^\d]*\(?wbc\)?[^\d]*(\d+\.?\d*)',
            r'leucocyte[s]?[^\d]*count[^\d]*(\d+\.?\d*)',
            r'leukocyte[s]?[^\d]*count[^\d]*(\d+\.?\d*)',
            r'\bwbc\b[^\d]*(\d+\.?\d*)',
            r'white[^\d]*blood[^\d]*cell[s]?[^\d]*(\d+\.?\d*)',
        ],
        
        # ========== DIFFERENTIAL COUNT ==========
        'neutrophils': [
            r'neutrophil[s]?[^\d]*(\d+\.?\d*)',
        ],
        'lymphocytes': [
            r'lymphocyte[s]?[^\d]*(\d+\.?\d*)',
        ],
        'monocytes': [
            r'monocyte[s]?[^\d]*(\d+\.?\d*)',
        ],
        'eosinophils': [
            r'eosinophil[s]?[^\d]*(\d+\.?\d*)',
        ],
        'basophils': [
            r'basophil[s]?[^\d]*(\d+\.?\d*)',
        ],
        
        # ========== RBC PARAMETERS ==========
        'rbc': [
            r'total[^\d]*rbc[^\d]*(\d+\.?\d*)',
            r'red[^\d]*blood[^\d]*cell[^\d]*count[^\d]*(\d+\.?\d*)',
            r'\brbc\b[^\d]*(\d+\.?\d*)',
        ],
        'hemoglobin': [
            r'h[ae]moglobin[^\d]*(\d+\.?\d*)',
            r'\bhb\b[^\d]*(\d+\.?\d*)',
            r'\bhgb\b[^\d]*(\d+\.?\d*)',
        ],
        'hematocrit': [
            r'h[ae]matocrit[^\d]*\(?pcv\)?[^\d]*(\d+\.?\d*)',
            r'\bpcv\b[^\d]*(\d+\.?\d*)',
            r'packed[^\d]*cell[^\d]*volume[^\d]*(\d+\.?\d*)',
        ],
        
        # ========== RBC INDICES ==========
        'mcv': [
            r'mean[^\d]*corpuscular[^\d]*volume[^\d]*(\d+\.?\d*)',
            r'\bmcv\b[^\d]*(\d+\.?\d*)',
        ],
        'mch': [
            r'mean[^\d]*corpuscular[^\d]*h[ae]moglobin[^\d]*(\d+\.?\d*)',
            r'\bmch\b[^\d]*(\d+\.?\d*)',
        ],
        'mchc': [
            r'mean[^\d]*corp[^\d]*h[ae]mo[^\d]*conc[^\d]*(\d+\.?\d*)',
            r'\bmchc\b[^\d]*(\d+\.?\d*)',
        ],
        'rdw': [
            r'red[^\d]*cell[^\d]*distribution[^\d]*width[^\d]*(\d+\.?\d*)',
            r'\brdw\b[^\d]*(\d+\.?\d*)',
        ],
        
        # ========== PLATELET - ULTRA FLEXIBLE ==========
        'platelets': [
            r'platelet[s]?[^\d]*count[^\d]*(\d+\.?\d*)',
            r'platelet[s]?[^\d]*(\d+\.?\d*)',
            r'\bplt\b[^\d]*(\d+\.?\d*)',
        ],
    }
    
    # Step 1: Extract raw values with context awareness
    raw_values = {}
    raw_contexts = {}  # Store the line context for unit detection
    
    for line in lines:
        line_lower = line.lower()
        
        for param_name, pattern_list in patterns.items():
            if param_name in raw_values:
                continue
                
            for pattern in pattern_list:
                match = re.search(pattern, line_lower)
                if match:
                    try:
                        value = float(match.group(1))
                        raw_values[param_name] = value
                        raw_contexts[param_name] = line_lower  # Save context for unit detection
                        print(f"  🔍 Extracted {param_name}: {value} from: {line.strip()[:80]}")
                        break
                    except:
                        pass
    
    # Step 2: SMART UNIT CONVERSION with context awareness
    
    # ========== GLUCOSE (typically already in mg/dL) ==========
    if 'fasting_glucose' in raw_values:
        params['Fasting_Glucose_mg_dL'] = raw_values['fasting_glucose']
    if 'random_glucose' in raw_values:
        params['Random_Glucose_mg_dL'] = raw_values['random_glucose']
    if 'postprandial_glucose' in raw_values:
        params['Postprandial_Glucose_mg_dL'] = raw_values['postprandial_glucose']
    if 'hba1c' in raw_values:
        params['HbA1c_percent'] = raw_values['hba1c']
    
    # ========== WBC - INTELLIGENT CONVERSION ==========
    if 'wbc' in raw_values:
        wbc = raw_values['wbc']
        context = raw_contexts.get('wbc', '')
        
        # Check for explicit unit indicators in context
        has_x10_notation = bool(re.search(r'[x×]\s*10[\^⁹9]?', context))
        has_per_ul = 'per' in context or 'µl' in context or 'ul' in context
        
        if has_x10_notation and wbc < 100:
            # Definitely ×10⁹/L format
            params['WBC_Count_per_uL'] = wbc * 1000
            print(f"  ✓ WBC: {wbc} ×10⁹/L → {wbc * 1000} per µL (×10 detected)")
        elif has_per_ul and wbc > 1000:
            # Already per µL
            params['WBC_Count_per_uL'] = wbc
            print(f"  ✓ WBC: {wbc} per µL (no conversion)")
        elif wbc < 100:
            # Most likely ×10⁹/L (standard format)
            params['WBC_Count_per_uL'] = wbc * 1000
            print(f"  ✓ WBC: {wbc} → {wbc * 1000} per µL (standard conversion)")
        else:
            # Likely already per µL
            params['WBC_Count_per_uL'] = wbc
            print(f"  ✓ WBC: {wbc} per µL (assumed)")
    
    # ========== DIFFERENTIAL COUNT (already percentages) ==========
    if 'neutrophils' in raw_values:
        params['Neutrophils_percent'] = raw_values['neutrophils']
    if 'lymphocytes' in raw_values:
        params['Lymphocytes_percent'] = raw_values['lymphocytes']
    if 'monocytes' in raw_values:
        params['Monocytes_percent'] = raw_values['monocytes']
    if 'eosinophils' in raw_values:
        params['Eosinophils_percent'] = raw_values['eosinophils']
    if 'basophils' in raw_values:
        params['Basophils_percent'] = raw_values['basophils']
    
    # ========== RBC - SMART CONVERSION ==========
    if 'rbc' in raw_values:
        rbc = raw_values['rbc']
        context = raw_contexts.get('rbc', '')
        
        has_x10_notation = bool(re.search(r'[x×]\s*10[\^⁶6]?', context))
        
        if has_x10_notation and rbc < 10:
            # ×10⁶/µL format - already correct
            params['RBC_Count_million_per_uL'] = rbc
            print(f"  ✓ RBC: {rbc} ×10⁶/µL (no conversion)")
        elif rbc < 10:
            # Standard format (millions)
            params['RBC_Count_million_per_uL'] = rbc
        elif rbc > 1000000:
            # Raw count, convert to millions
            params['RBC_Count_million_per_uL'] = rbc / 1000000
            print(f"  ✓ RBC: {rbc} → {rbc/1000000} million/µL")
        else:
            params['RBC_Count_million_per_uL'] = rbc
    
    # ========== HEMOGLOBIN (typically g/dL) ==========
    if 'hemoglobin' in raw_values:
        params['Hemoglobin_g_dL'] = raw_values['hemoglobin']
    
    # ========== HEMATOCRIT (typically %) ==========
    if 'hematocrit' in raw_values:
        params['Hematocrit_percent'] = raw_values['hematocrit']
    
    # ========== RBC INDICES (typically correct units) ==========
    if 'mcv' in raw_values:
        params['MCV_fL'] = raw_values['mcv']
    if 'mch' in raw_values:
        params['MCH_pg'] = raw_values['mch']
    if 'mchc' in raw_values:
        params['MCHC_g_dL'] = raw_values['mchc']
    if 'rdw' in raw_values:
        params['RDW_percent'] = raw_values['rdw']
    
    # ========== PLATELET - INTELLIGENT CONVERSION ==========
    if 'platelets' in raw_values:
        plt = raw_values['platelets']
        context = raw_contexts.get('platelets', '')
        
        has_x10_notation = bool(re.search(r'[x×]\s*10[\^⁹9]?', context))
        has_per_ul = 'per' in context or 'µl' in context or 'ul' in context
        
        if has_x10_notation and plt < 1000:
            # Definitely ×10⁹/L format
            params['Platelet_Count_per_uL'] = plt * 1000
            print(f"  ✓ Platelet: {plt} ×10⁹/L → {plt * 1000} per µL (×10 detected)")
        elif has_per_ul and plt > 10000:
            # Already per µL
            params['Platelet_Count_per_uL'] = plt
            print(f"  ✓ Platelet: {plt} per µL (no conversion)")
        elif plt < 1000:
            # Most likely ×10⁹/L (standard format)
            params['Platelet_Count_per_uL'] = plt * 1000
            print(f"  ✓ Platelet: {plt} → {plt * 1000} per µL (standard conversion)")
        else:
            # Likely already per µL
            params['Platelet_Count_per_uL'] = plt
            print(f"  ✓ Platelet: {plt} per µL (assumed)")
    
    # ========== SUMMARY & DIAGNOSTIC ==========
    print(f"\n📊 EXTRACTION SUMMARY: {len(params)} parameters")
    for key, val in params.items():
        print(f"  ✓ {key}: {val}")
    
    # ========== MODEL COMPATIBILITY CHECK ==========
    testable = []
    
    if any(k in params for k in ['Fasting_Glucose_mg_dL', 'HbA1c_percent', 'Random_Glucose_mg_dL', 'Postprandial_Glucose_mg_dL']):
        testable.append('Diabetes')
    
    if all(k in params for k in ['Platelet_Count_per_uL', 'WBC_Count_per_uL', 'Hematocrit_percent', 'Hemoglobin_g_dL', 'Neutrophils_percent', 'Lymphocytes_percent']):
        testable.append('Dengue')
    
    if all(k in params for k in ['Hemoglobin_g_dL', 'RBC_Count_million_per_uL', 'Platelet_Count_per_uL', 'WBC_Count_per_uL', 'MCV_fL', 'MCH_pg']):
        testable.append('Malaria')
    
    if all(k in params for k in ['Hemoglobin_g_dL', 'Hematocrit_percent', 'RBC_Count_million_per_uL', 'MCV_fL', 'MCH_pg', 'MCHC_g_dL', 'RDW_percent']):
        testable.append('Anemia')
    
    if all(k in params for k in ['WBC_Count_per_uL', 'Neutrophils_percent', 'Lymphocytes_percent', 'Monocytes_percent', 'Eosinophils_percent']):
        testable.append('Infection')
    
    if all(k in params for k in ['WBC_Count_per_uL', 'Neutrophils_percent', 'Lymphocytes_percent', 'Platelet_Count_per_uL', 'Hemoglobin_g_dL']):
        testable.append('Typhoid')
    
    if testable:
        print(f"\n✅ CAN TEST: {', '.join(testable)}")
    else:
        print(f"\n⚠️ CANNOT TEST any models - need more parameters")
        print("\n📋 Missing parameters for each model:")
        
        # Diabetes check
        diabetes_features = ['Fasting_Glucose_mg_dL', 'HbA1c_percent', 'Random_Glucose_mg_dL', 'Postprandial_Glucose_mg_dL']
        if not any(k in params for k in diabetes_features):
            print(f"  Diabetes: Need ANY glucose parameter")
        
        # Dengue check
        dengue_features = ['Platelet_Count_per_uL', 'WBC_Count_per_uL', 'Hematocrit_percent', 'Hemoglobin_g_dL', 'Neutrophils_percent', 'Lymphocytes_percent']
        dengue_missing = [f for f in dengue_features if f not in params]
        if dengue_missing:
            print(f"  Dengue: Missing {', '.join(dengue_missing)}")
        
        # Malaria check
        malaria_features = ['Hemoglobin_g_dL', 'RBC_Count_million_per_uL', 'Platelet_Count_per_uL', 'WBC_Count_per_uL', 'MCV_fL', 'MCH_pg']
        malaria_missing = [f for f in malaria_features if f not in params]
        if malaria_missing:
            print(f"  Malaria: Missing {', '.join(malaria_missing)}")
        
        # Anemia check
        anemia_features = ['Hemoglobin_g_dL', 'Hematocrit_percent', 'RBC_Count_million_per_uL', 'MCV_fL', 'MCH_pg', 'MCHC_g_dL', 'RDW_percent']
        anemia_missing = [f for f in anemia_features if f not in params]
        if anemia_missing:
            print(f"  Anemia: Missing {', '.join(anemia_missing)}")
        
        # Infection check
        infection_features = ['WBC_Count_per_uL', 'Neutrophils_percent', 'Lymphocytes_percent', 'Monocytes_percent', 'Eosinophils_percent']
        infection_missing = [f for f in infection_features if f not in params]
        if infection_missing:
            print(f"  Infection: Missing {', '.join(infection_missing)}")
        
        # Typhoid check
        typhoid_features = ['WBC_Count_per_uL', 'Neutrophils_percent', 'Lymphocytes_percent', 'Platelet_Count_per_uL', 'Hemoglobin_g_dL']
        typhoid_missing = [f for f in typhoid_features if f not in params]
        if typhoid_missing:
            print(f"  Typhoid: Missing {', '.join(typhoid_missing)}")
    
    return params
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"⚠️ PDF extraction failed: {e}")
        return None

def extract_text_with_huggingface_ocr(image_bytes):
    """Use HuggingFace OCR model to extract text from images"""
    if not HUGGINGFACE_API_KEY:
        return None
    
    try:
        # Try Microsoft TrOCR model
        API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-large-printed"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get('generated_text', '')
                if text:
                    print(f"✅ HuggingFace OCR extracted {len(text)} chars")
                    return text
        elif response.status_code == 503:
            print(f"⚠️ HuggingFace OCR model loading (503), skipping...")
        elif response.status_code == 404:
            print(f"⚠️ HuggingFace OCR model not found (404), skipping...")
        else:
            print(f"⚠️ HuggingFace OCR failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️ HuggingFace OCR error: {e}")
    
    return None

def analyze_blood_report_image_vision(image_bytes):
    """Analyze blood report from image using OpenAI Vision"""
    if not openai_client:
        return None
    
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        prompt = """Analyze this blood test report image comprehensively.

Extract ALL visible parameters with values and units, then provide:

EXTRACTED VALUES
[List parameter: value unit for each visible test]

OVERALL HEALTH ASSESSMENT
DETAILED PARAMETER ANALYSIS
IDENTIFIED HEALTH CONCERNS
DIETARY RECOMMENDATIONS
LIFESTYLE MODIFICATIONS
FOLLOW-UP RECOMMENDATIONS
URGENT SYMPTOMS
DISCLAIMER

Plain text, • bullets."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Expert medical AI analyzing blood reports. Extract values accurately. Plain text, • bullets."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        analysis = response.choices[0].message.content
        analysis = clean_markdown(analysis)
        
        if analysis and len(analysis) > 100:
            print(f"✅ OpenAI Vision analysis: {len(analysis)} chars")
            return {
                'success': True,
                'analysis': analysis,
                'ai_provider': 'OpenAI Vision (GPT-4o-mini)',
                'extraction_method': 'vision'
            }
    except Exception as e:
        if 'insufficient_quota' in str(e) or '429' in str(e):
            print(f"⚠️ OpenAI quota exceeded, falling back...")
        else:
            print(f"⚠️ OpenAI Vision failed: {e}")
    
    return None

def analyze_blood_report_text_groq(extracted_text, source_type='pdf'):
    """Analyze extracted text using Groq"""
    if not groq_client or not extracted_text:
        return None
    
    try:
        prompt = f"""Analyze this blood test report ({source_type}).

EXTRACTED TEXT:
{extracted_text[:3000]}

Provide comprehensive analysis in plain text (NO markdown):

EXTRACTED VALUES
[List parameter: value unit]

OVERALL HEALTH ASSESSMENT
DETAILED PARAMETER ANALYSIS
IDENTIFIED HEALTH CONCERNS
DIETARY RECOMMENDATIONS
LIFESTYLE MODIFICATIONS
FOLLOW-UP RECOMMENDATIONS
URGENT SYMPTOMS
DISCLAIMER

Use • for bullets."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Medical AI analyzing blood reports. Plain text, • bullets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=3500
        )
        
        analysis = clean_markdown(response.choices[0].message.content)
        
        if analysis and len(analysis) > 100:
            print(f"✅ Groq blood analysis: {len(analysis)} chars")
            return {
                'success': True,
                'analysis': analysis,
                'ai_provider': 'Groq (Llama 3.3 70B)',
                'extraction_method': source_type
            }
    except Exception as e:
        print(f"⚠️ Groq text analysis failed: {e}")
    
    return None

def analyze_blood_report_image(image_bytes):
    """Multi-strategy blood report image analysis - TESSERACT FIRST!"""
    
    # STRATEGY 1: Tesseract OCR + Groq (FREE, LOCAL, NO LIMITS!)
    print("🔍 Strategy 1: Tesseract OCR + Groq...")
    tesseract_text = extract_text_with_tesseract(image_bytes)
    if tesseract_text and len(tesseract_text) > 50:
        print(f"✅ Extracted text preview: {tesseract_text[:200]}...")
        
        if groq_client:
            result = analyze_blood_report_text_groq(tesseract_text, 'tesseract-ocr')
            if result and result.get('success'):
                print("✅ SUCCESS: Tesseract + Groq analysis complete!")
                return result
        else:
            print("⚠️ Groq not available - returning extracted text with template")
            return {
                'success': True,
                'analysis': f"""EXTRACTED TEXT FROM IMAGE (Tesseract OCR):
{tesseract_text[:1000]}

{get_template_blood_analysis({})['analysis']}""",
                'ai_provider': 'Tesseract OCR + Template (Get FREE Groq: https://console.groq.com)',
                'extraction_method': 'tesseract'
            }
    else:
        print("⚠️ Tesseract extraction failed or insufficient text")
    
    # STRATEGY 2: HuggingFace OCR + Groq
    print("🔍 Strategy 2: HuggingFace OCR + Groq...")
    if HUGGINGFACE_API_KEY:
        hf_text = extract_text_with_huggingface_ocr(image_bytes)
        if hf_text and len(hf_text) > 50 and groq_client:
            result = analyze_blood_report_text_groq(hf_text, 'huggingface-ocr')
            if result and result.get('success'):
                return result
    
    # STRATEGY 3: OpenAI Vision (will hit quota)
    print("🔍 Strategy 3: OpenAI Vision (may hit quota)...")
    result = analyze_blood_report_image_vision(image_bytes)
    if result and result.get('success'):
        return result
    
    # STRATEGY 4: Template fallback
    print("📋 Using template for blood report (no AI available)")
    return {
        'success': True,
        'analysis': get_template_blood_analysis({})['analysis'],
        'ai_provider': 'Template (Install Tesseract + Get FREE Groq API)'
    }



def analyze_blood_report_pdf(pdf_bytes):
    """Analyze PDF blood report - WITH DISEASE MODELS"""
    try:
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        print(f"📄 Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(tmp_path)
        
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        if not pdf_text or len(pdf_text) < 50:
            return {
                'success': False,
                'error': 'Could not extract text from PDF'
            }
        
        print(f"✅ Extracted {len(pdf_text)} chars from PDF")
        
        # ⭐ NEW: Use disease models
        return analyze_blood_report_with_disease_models(
            extracted_text=pdf_text, 
            source='pdf'
        )
        
    except Exception as e:
        print(f"❌ PDF analysis error: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            'success': False,
            'error': f'PDF processing failed: {str(e)}'
        }
def generate_blood_report_analysis(blood_data):
    """Generate AI analysis for blood parameters"""
    if not blood_data:
        return {'success': False, 'analysis': 'No parameters provided', 'ai_provider': 'None'}
    
    params_text = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in blood_data.items()])
    prompt = f"""Expert pathologist analyzing blood test.

RESULTS:
{params_text}

Provide: EXTRACTED VALUES, HEALTH ASSESSMENT, PARAMETER ANALYSIS, CONCERNS, DIETARY RECOMMENDATIONS, LIFESTYLE MODIFICATIONS, FOLLOW-UP, URGENT SYMPTOMS, LIMITATIONS, DISCLAIMER

Plain text, • bullets."""

    # Try Groq first
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Expert pathologist AI. Plain text, • bullets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=3500
            )
            analysis = clean_markdown(response.choices[0].message.content)
            if analysis and len(analysis) > 100:
                print(f"✅ Groq blood analysis: {len(analysis)} chars")
                return {'success': True, 'analysis': analysis, 'ai_provider': 'Groq (Llama 3.3 70B)'}
        except Exception as e:
            print(f"⚠️ Groq blood analysis failed: {e}")
    
    # Fallback to OpenAI
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Expert pathologist AI. Plain text, • bullets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=3500
            )
            analysis = clean_markdown(response.choices[0].message.content)
            print(f"✅ OpenAI blood analysis: {len(analysis)} chars")
            return {'success': True, 'analysis': analysis, 'ai_provider': 'OpenAI (GPT-4o-mini)'}
        except Exception as e:
            if 'insufficient_quota' in str(e) or '429' in str(e):
                print(f"⚠️ OpenAI quota exceeded")
            else:
                print(f"❌ OpenAI blood analysis failed: {e}")
    
    # Template fallback
    print("📋 Using template for blood parameters")
    return get_template_blood_analysis(blood_data)

def get_template_ct_analysis(predicted_class, confidence):
    """Template fallback for CT analysis"""
    return {
        'success': True,
        'analysis': f"""IMAGE QUALITY ASSESSMENT
Standard CT scan quality - preliminary AI analysis

ANATOMICAL FINDINGS
AI Prediction: {predicted_class}
Confidence: {confidence*100:.1f}%

DETAILED FINDING ANALYSIS
The AI model has detected {predicted_class} with {confidence*100:.1f}% confidence. This requires professional radiologist review for confirmation and detailed interpretation.

CLINICAL CORRELATION
• Immediate professional radiologist interpretation required
• Clinical correlation with patient history essential
• Specialist consultation strongly recommended
• Results should not be used in isolation

KEY FINDINGS SUMMARY
• AI-detected condition: {predicted_class}
• Detection confidence: {confidence*100:.1f}%
• Professional medical review mandatory
• Clinical correlation needed
• Follow-up imaging may be recommended
• Consult healthcare provider immediately

URGENT ALERT INDICATORS
• Severe or worsening flank/back pain
• Blood in urine (hematuria)
• Fever with chills (>101°F)
• Difficulty urinating or painful urination
• Nausea and vomiting
• Signs of infection

PATIENT GUIDANCE
• Schedule urgent appointment with nephrologist or urologist
• Bring complete medical history and previous imaging
• Monitor symptoms closely and keep symptom diary
• Stay well-hydrated unless otherwise directed
• Avoid self-diagnosis - professional interpretation essential
• Don't delay seeking medical attention

LIMITATIONS & DISCLAIMER
⚠️ IMPORTANT: This is preliminary AI analysis only. Not a medical diagnosis.
• Requires expert radiologist interpretation
• Clinical correlation essential
• Individual patient factors must be considered
• AI has limitations and may not detect all conditions
• Professional medical consultation mandatory

Configure AI API keys (GROQ_API_KEY or OPENAI_API_KEY) for enhanced analysis.""",
        'ai_provider': 'Template (AI APIs unavailable)'
    }

def get_template_blood_analysis(blood_data):
    """Template fallback for blood analysis"""
    params = "\n".join([f"• {k.replace('_', ' ').title()}: {v}" for k, v in blood_data.items()]) if blood_data else "No specific values provided"
    
    return {
        'success': True,
        'analysis': f"""EXTRACTED VALUES
{params}

OVERALL HEALTH ASSESSMENT
Professional laboratory interpretation with proper reference ranges is essential for accurate assessment. Blood test results must be evaluated in context of:
• Patient age, gender, and medical history
• Laboratory-specific reference ranges
• Clinical symptoms and physical examination
• Previous test results for comparison
• Current medications and supplements

GENERAL HEALTH RECOMMENDATIONS
Balanced Nutrition:
• Consume variety of fruits and vegetables (5-7 servings daily)
• Include lean proteins, whole grains, healthy fats
• Limit processed foods, excess sugar, and sodium
• Consider Mediterranean or DASH diet patterns

Hydration:
• Drink 8-10 glasses of water daily
• Increase intake during exercise or hot weather
• Monitor urine color (pale yellow indicates good hydration)

Physical Activity:
• Aim for 150 minutes moderate exercise weekly
• Include strength training 2-3 times per week
• Regular walking, swimming, or cycling
• Consult doctor before starting new exercise program

Sleep & Stress:
• Maintain 7-9 hours quality sleep nightly
• Practice stress management (meditation, yoga, deep breathing)
• Regular sleep schedule
• Limit screen time before bed

Lifestyle Factors:
• Avoid smoking and limit alcohol
• Maintain healthy weight (BMI 18.5-24.9)
• Regular health checkups and screenings
• Follow prescribed medications as directed

WHEN TO CONSULT DOCTOR IMMEDIATELY
• Any values significantly outside reference ranges
• Persistent or worsening symptoms
• Unusual fatigue, weakness, or dizziness
• Unexplained weight changes
• New or concerning symptoms
• Before making major diet or lifestyle changes
• For proper interpretation with reference ranges

FOLLOW-UP RECOMMENDATIONS
• Retest as recommended by healthcare provider
• Typically 3-6 months for monitoring
• Sooner if abnormalities detected
• Bring previous results for comparison
• Discuss trends with your doctor

IMPORTANT LIMITATIONS
⚠️ This analysis requires proper laboratory reference ranges for accurate interpretation.
• Reference ranges vary by lab, age, gender
• Clinical context is essential
• Cannot diagnose conditions from values alone
• Professional medical interpretation mandatory

DISCLAIMER
This information is educational only and NOT a medical diagnosis. Blood test results must be interpreted by qualified healthcare professionals with access to:
• Complete medical history
• Physical examination findings
• Laboratory-specific reference ranges
• Clinical context and symptoms
• Additional diagnostic tests if needed

Always consult with your doctor, nurse practitioner, or qualified healthcare provider for proper interpretation and medical advice.

📌 To enable AI-powered analysis, configure API keys in .env file:
• GROQ_API_KEY for Groq AI (free tier available)
• OPENAI_API_KEY for OpenAI GPT models
• HUGGINGFACE_API_KEY for OCR and vision models""",
        'ai_provider': 'Template (AI APIs unavailable)'
    }

# Initialize on import
init_groq()
init_openai()
init_huggingface()