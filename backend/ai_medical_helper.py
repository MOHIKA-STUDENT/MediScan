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
ollama_available = False
OLLAMA_MODELS = []  # Will be populated by init_ollama
OLLAMA_URL = 'http://localhost:11434/api/generate'




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

def init_ollama():
    """Check if Ollama is running and discover all available models"""
    global ollama_available, OLLAMA_MODELS
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            all_names = [m['name'] for m in models]
            # Priority order for medical analysis quality
            preferred_order = ['phi3', 'mistral', 'llama3', 'ats_llama3']
            OLLAMA_MODELS = []
            # Add preferred models first (in priority order)
            for pref in preferred_order:
                for name in all_names:
                    if name.split(':')[0] == pref and name not in OLLAMA_MODELS:
                        OLLAMA_MODELS.append(name)
            # Add any remaining models
            for name in all_names:
                if name not in OLLAMA_MODELS:
                    OLLAMA_MODELS.append(name)
            if OLLAMA_MODELS:
                print(f"✅ Ollama available. Models ready: {OLLAMA_MODELS}")
                ollama_available = True
                return True
    except Exception as e:
        print(f"⚠️ Ollama not available: {e}")
    ollama_available = False
    return False


def call_ollama_single(model_name, prompt, system_prompt, result_container, idx):
    """Call a single Ollama model — runs in thread for parallel execution"""
    try:
        full_prompt = f"{system_prompt}\n\n{prompt}"
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                   "num_predict": 800,
                    "num_ctx": 4096
                }
            },
            timeout=300
        )
        if response.status_code == 200:
            text = response.json().get('response', '').strip()
            if text and len(text) > 100:
                result_container[idx] = {
                    'text': clean_markdown(text),
                    'model': model_name,
                    'success': True
                }
                print(f"✅ Ollama {model_name} responded: {len(text)} chars")
                return
        result_container[idx] = {'success': False, 'model': model_name}
    except Exception as e:
        print(f"⚠️ Ollama {model_name} failed: {e}")
        result_container[idx] = {'success': False, 'model': model_name}


def call_ollama_fastest(prompt, system_prompt="You are an expert medical doctor providing professional analysis. Use plain text with • for bullet points."):
    """
    Call ALL available Ollama models in parallel.
    Returns the FASTEST successful response.
    No API limits. No cost. Fully local.
    """
    global ollama_available
    if not ollama_available or not OLLAMA_MODELS:
        return None

    import threading

    models_to_try = OLLAMA_MODELS[:1]  # Only fastest model — phi3  # Max 3 parallel (avoid overwhelming RAM)
    result_container = [None] * len(models_to_try)
    winner = [None]  # Shared winner slot
    winner_lock = threading.Lock()
    done_event = threading.Event()

    def run_model(model, idx):
        call_ollama_single(model, prompt, system_prompt, result_container, idx)
        with winner_lock:
            if winner[0] is None and result_container[idx] and result_container[idx].get('success'):
                winner[0] = result_container[idx]
                done_event.set()

    threads = []
    for i, model in enumerate(models_to_try):
        t = threading.Thread(target=run_model, args=(model, i), daemon=True)
        threads.append(t)
        t.start()
        print(f"🚀 Launched Ollama {model} (parallel)")

    # Wait for first success or all threads to finish (max 130s)
    done_event.wait(timeout=130)

    if winner[0]:
        print(f"🏆 Fastest Ollama response: {winner[0]['model']}")
        return winner[0]['text'], winner[0]['model']

    # If parallel failed, check all results for any success
    for r in result_container:
        if r and r.get('success'):
            return r['text'], r['model']

    print("⚠️ All Ollama models failed or timed out")
    ollama_available = False
    return None

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
        if '429' in str(e) or 'rate_limit' in str(e).lower():
            print("🔄 Groq rate limited — racing all Ollama models for CT...")
            ollama_result = call_ollama_fastest(
                prompt,
                "You are an expert radiologist. Provide detailed kidney CT scan analysis in plain text with • bullets."
            )
            if ollama_result:
                text, model_name = ollama_result
                return {
                    'success': True,
                    'analysis': text,
                    'ai_provider': f'Ollama {model_name} [local radiologist]'
                }
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
        print(f"⚠️ Only {len(blood_params) if blood_params else 0} parameters extracted — using full text AI analysis")
        if groq_client and extracted_text:
            result = analyze_blood_report_text_groq(extracted_text, source)
            if result and result.get('success'):
                return result
        if extracted_text:
            return get_template_blood_analysis(blood_params or {})
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
            prompt = f"""You are an expert pathologist. The user has uploaded a real blood test report. Speak directly to them.

ACTUAL BLOOD VALUES FROM THEIR REPORT:
{params_text}

ML DISEASE SCREENING RESULTS (6 RandomForest models):
{predictions_text}

POSITIVE FINDINGS: {positive_text}

Write a full medical report in plain text (NO markdown, use • for bullets). Base EVERYTHING on the actual values above — do not use placeholder ranges or generic statements. Every number you cite must come from the values listed above.

DIAGNOSED CONDITIONS
- For each positive disease: state it clearly (e.g. "Based on your report, you are likely experiencing Dengue fever"). Explain which specific values from YOUR report triggered this (e.g. "Your platelets of 45,000/µL are critically low — normal is 150,000–400,000/µL, and values below 100,000 are a hallmark of Dengue"). Include confidence level. Rate severity: Low / Medium / High.
- If no positives: state clearly "No major diseases detected. Your values are within acceptable ranges overall."

EXTRACTED VALUES ANALYSIS
- List every parameter extracted: name, value with unit, status (Normal/High/Low), and the normal reference range.
- Flag any value that is significantly abnormal and explain what it means clinically.

OVERALL HEALTH ASSESSMENT
- Summarise the overall picture based on the actual values. Be specific — mention actual numbers.
- Overall severity: Low / Medium / High.

DISEASE-SPECIFIC GUIDANCE (only for detected positives)
- Symptoms to expect, immediate steps, treatment, prognosis.

DIETARY RECOMMENDATIONS
- Specific to the detected findings. If Dengue: fluids, vitamin C, avoid aspirin. If Anaemia: iron-rich foods. Etc.

FOLLOW-UP RECOMMENDATIONS
- Specific tests needed, timeframe, specialist to see.

URGENT SYMPTOMS TO WATCH
- Red flags for the detected conditions requiring emergency care.

LIMITATIONS & DISCLAIMER
- This is AI/ML-assisted screening. Not a clinical diagnosis. See a qualified doctor for confirmation."""

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
            if '429' in str(e) or 'rate_limit' in str(e).lower():
                print("🔄 Groq rate limited — racing all Ollama models for ML+AI...")
                ollama_result = call_ollama_fastest(
                    prompt,
                    "You are an expert pathologist. Analyze blood reports directly and empathetically. Plain text, • bullets."
                )
                if ollama_result:
                    text, model_name = ollama_result
                    return {
                        'success': True,
                        'disease_predictions': disease_results,
                        'analysis': text,
                        'ai_provider': f'ML Disease Models (6) + Ollama {model_name} [local — fastest]',
                        'extraction_method': source,
                        'has_disease_classification': True,
                        'num_positive_diseases': disease_results['num_positive'],
                        'positive_diseases': disease_results['positive_diseases'],
                        'detected_diseases_summary': f"ML screening complete. Positive: {positive_text}"
                    }
      # All cloud AI failed — try Ollama directly with full text
    print("🔄 All cloud AI failed — trying Ollama with full report text...")
    if ollama_available and extracted_text:
        ollama_prompt = f"""You are an expert pathologist. Analyze this real blood test report and speak directly to the patient.

BLOOD VALUES:
{chr(10).join([f'• {k}: {v}' for k, v in blood_params.items()])}

ML DISEASE SCREENING:
{chr(10).join([f'• {d}: {"POSITIVE" if p.get("is_positive") else "Negative"} ({p.get("confidence",0)*100:.0f}%)' for d, p in disease_results["predictions"].items() if "error" not in p])}

DETECTED CONDITIONS: {", ".join(disease_results["positive_diseases"]) if disease_results["positive_diseases"] else "None"}

Write a clear medical report. Plain text, • bullets. Cover:
DIAGNOSED CONDITIONS (explain what values caused each positive finding)
EXTRACTED VALUES ANALYSIS (list each value, normal range, status)
OVERALL HEALTH ASSESSMENT
DIETARY RECOMMENDATIONS
FOLLOW-UP RECOMMENDATIONS
URGENT SYMPTOMS TO WATCH
DISCLAIMER"""

        ollama_result = call_ollama_fastest(
            ollama_prompt,
            "You are an expert pathologist. Be direct, specific, use actual numbers. Plain text, • bullets only."
        )
        if ollama_result:
            text, model_name = ollama_result
            return {
                'success': True,
                'disease_predictions': disease_results,
                'analysis': text,
                'ai_provider': f'ML Disease Models (6) + Ollama {model_name} [local]',
                'extraction_method': source,
                'has_disease_classification': True,
                'num_positive_diseases': disease_results['num_positive'],
                'positive_diseases': disease_results['positive_diseases'],
                'detected_diseases_summary': f"ML screening: {', '.join(disease_results['positive_diseases']) if disease_results['positive_diseases'] else 'No major diseases detected'}"
            }

    # Absolute last resort — structured template with actual values (NOT generic)
    print("📋 All AI failed — using structured value summary")
    positive_diseases_local = disease_results.get('positive_diseases', [])

    # Build a meaningful non-generic summary using actual values
    value_lines = "\n".join([f"• {k.replace('_', ' ')}: {v}" for k, v in blood_params.items()])
    disease_lines = "\n".join([
        f"• {d}: {'⚠️ POSITIVE' if p.get('is_positive') else '✓ Negative'} ({p.get('confidence', 0)*100:.0f}% confidence)"
        for d, p in disease_results['predictions'].items() if 'error' not in p
    ])

    if positive_diseases_local:
        condition_summary = f"⚠️ The following conditions were detected: {', '.join(positive_diseases_local)}. Please consult a doctor immediately for confirmation and treatment."
    else:
        condition_summary = "✓ No major diseases detected by ML screening. Your values appear within acceptable ranges overall."

    structured_analysis = f"""DIAGNOSED CONDITIONS
{condition_summary}

ML DISEASE SCREENING RESULTS
{disease_lines}

YOUR BLOOD VALUES
{value_lines}

OVERALL HEALTH ASSESSMENT
Based on your extracted values above, the ML models have completed screening. {condition_summary}

FOLLOW-UP RECOMMENDATIONS
- Consult a general physician or specialist within 24-48 hours if any positive findings
- Bring this report and all values listed above to your appointment
- Repeat CBC in 2-4 weeks to monitor trends

URGENT SYMPTOMS TO WATCH
- High fever (>101°F/38.3°C), chills, or rigors
- Severe fatigue or weakness
- Unusual bruising or bleeding
- Difficulty breathing
- Seek emergency care immediately if any of these develop

DISCLAIMER
- This is ML-assisted screening only — not a clinical diagnosis
- See a qualified doctor for proper interpretation
- AI summarization was unavailable — showing structured ML results"""

    return {
        'success': True,
        'disease_predictions': disease_results,
        'analysis': structured_analysis,
        'ai_provider': 'ML Disease Models (6) + Structured Summary [AI unavailable]',
        'has_disease_classification': True,
        'num_positive_diseases': disease_results['num_positive'],
        'positive_diseases': disease_results['positive_diseases']
    }

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
    """Extract text from PDF using pdfplumber (better than PyPDF2 for complex PDFs)"""
    text = ""
    
    # Try pdfplumber first (handles complex layouts better)
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip() and len(text.strip()) > 100:
            print(f"✅ pdfplumber extracted {len(text.strip())} chars")
            return text.strip()
    except Exception as e:
        print(f"⚠️ pdfplumber failed: {e}")
    
    # Fallback to PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            print(f"✅ PyPDF2 extracted {len(text.strip())} chars")
            return text.strip()
    except Exception as e:
        print(f"⚠️ PyPDF2 failed: {e}")
    
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
    """Analyze extracted text using Groq — handles comprehensive reports"""
    if not groq_client or not extracted_text:
        return None
    
    try:
        # For large reports, send in chunks or summarize key sections
        # Find the most relevant section (tests outside reference range + key values)
        text_to_analyze = extracted_text
        
        # If text is very long, prioritize the abnormal values section + hemogram
        if len(extracted_text) > 4000:
            # Try to find the "Tests Outside Reference Range" section
            lower = extracted_text.lower()
            outside_range_idx = lower.find('outside reference range')
            hemogram_idx = lower.find('hemogram')
            
            if outside_range_idx > 0:
                # Start from the abnormal values section
                text_to_analyze = extracted_text[max(0, outside_range_idx-200):outside_range_idx+3000]
                # Also include the full hemogram if found separately
                if hemogram_idx > 0 and hemogram_idx > outside_range_idx + 3000:
                    text_to_analyze += "\n\n" + extracted_text[hemogram_idx:hemogram_idx+2000]
            else:
                text_to_analyze = extracted_text[:5000]
        
        prompt = f"""You are an expert pathologist analyzing a real blood test report. Speak directly to the patient.

EXTRACTED REPORT TEXT:
{text_to_analyze}

This is a REAL patient report. Extract ALL test values you can find and provide a comprehensive analysis.

Write in plain text (NO markdown). Use • for bullets.

DIAGNOSED CONDITIONS
- Based on the values in this report, clearly state any detected health concerns with specific numbers from the report.
- If values are normal overall, say so clearly.

EXTRACTED VALUES ANALYSIS
- List EVERY test result you can identify: test name, value, unit, normal range, and status (Normal/High/Low).
- Pay special attention to values marked as outside reference range in the report.

OVERALL HEALTH ASSESSMENT
- Summarize the patient's overall health picture. Mention actual numbers.
- Note the most critical findings first.

DETAILED PARAMETER ANALYSIS  
- Explain what each abnormal value means clinically.
- Connect related findings (e.g. low vitamin D + fatigue).

IDENTIFIED HEALTH CONCERNS
- List specific concerns with supporting values from the report.

DIETARY RECOMMENDATIONS
- Specific to the findings. If low Vitamin D: sun exposure, supplements, diet. If high Prolactin: stress management, etc.

LIFESTYLE MODIFICATIONS
- Based on actual findings in this report.

FOLLOW-UP RECOMMENDATIONS  
- Which specialist to see, which tests to repeat, timeframe.

URGENT SYMPTOMS TO WATCH
- Red flags specific to the abnormal values found.

LIMITATIONS & DISCLAIMER
- This is AI-assisted analysis. Not a clinical diagnosis. See a qualified doctor."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Expert pathologist analyzing real patient blood reports. Extract ALL values from the text. Be specific with actual numbers. Plain text, • bullets. Never use generic advice — always reference the actual values from the report."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4500
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
        if '429' in str(e) or 'rate_limit' in str(e).lower():
            print("🔄 Groq rate limited — trying Ollama with condensed prompt...")
            # Use shorter prompt for Ollama (faster)
            short_prompt = f"""Analyze this blood report. Be specific with numbers. Plain text, • bullets.

{extracted_text[:1500]}

Cover: DIAGNOSED CONDITIONS, KEY ABNORMAL VALUES, HEALTH ASSESSMENT, DIETARY ADVICE, FOLLOW-UP, DISCLAIMER"""
            ollama_result = call_ollama_fastest(
                short_prompt,
                "Expert pathologist. Specific, concise. Plain text, • bullets."
            )
            if ollama_result:
                text, model_name = ollama_result
                return {
                    'success': True,
                    'analysis': text,
                    'ai_provider': f'Ollama {model_name} [local]',
                    'extraction_method': source_type
                }
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
...
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

""",
      
    }


# ============================================
# VALIDATION FUNCTIONS
# ============================================

def validate_ct_scan(image_bytes):
    """
    Check if uploaded image is a kidney CT scan.
    Uses OpenAI Vision if available, otherwise strict grayscale + metadata check.
    Returns: {'is_valid': bool, 'reason': str}
    """
    # Strategy 1: OpenAI Vision (most accurate)
    if openai_client:
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            prompt = """Look at this image carefully. Is this a medical CT scan (computed tomography) image of a kidney or abdomen?

CT scans have these characteristics:
- Grayscale/black and white only
- Shows internal body structures (organs, bones)
- Has DICOM-style overlays (patient info, scan parameters like kV, mA, slice thickness)
- Has a black background with gray anatomical structures
- No color, no natural scene, no faces, no text documents

Reply with EXACTLY one word only:
VALID - if this is clearly a CT scan of kidney/abdomen
INVALID - if this is anything else (photo, blood report, MRI, chest xray, natural image, etc.)"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=5,
                temperature=0
            )
            answer = response.choices[0].message.content.strip().upper()
            print(f"🔍 CT Validation (OpenAI): {answer}")
            if "VALID" in answer and "INVALID" not in answer:
                return {'is_valid': True, 'reason': 'Confirmed kidney CT scan'}
            else:
                return {'is_valid': False, 'reason': 'This image does not appear to be a kidney CT scan. Please upload a JPEG or PNG of a CT scan of the kidney/abdomen.'}
        except Exception as e:
            print(f"⚠️ OpenAI CT validation error: {e}")
            # Don't fall through to permissive fallback — use strict checks below

    # Strategy 2: Groq cannot see images — use STRICT image analysis only
    # Check 1: Must be grayscale (CT scans are always grayscale)
    try:
        from PIL import Image as PILImage
        img = PILImage.open(BytesIO(image_bytes))
        img_array = np.array(img)

        is_grayscale = False

        if len(img_array.shape) == 2:
            is_grayscale = True
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            is_grayscale = True
        elif len(img_array.shape) == 3:
            r = img_array[:,:,0].astype(float)
            g = img_array[:,:,1].astype(float)
            b = img_array[:,:,2].astype(float)
            rg_diff = np.mean(np.abs(r - g))
            rb_diff = np.mean(np.abs(r - b))
            gb_diff = np.mean(np.abs(g - b))
            # Very strict: all channels must be nearly identical
            is_grayscale = (rg_diff < 3 and rb_diff < 3 and gb_diff < 3)

        if not is_grayscale:
            print(f"❌ CT Validation: Image has color channels — not a CT scan")
            return {'is_valid': False, 'reason': 'This does not appear to be a CT scan. CT scans are grayscale. Please upload a kidney CT scan image.'}

        # Check 2: Image must be reasonably large (CT scans are typically 512x512 or larger)
        h, w = img_array.shape[:2]
        if h < 200 or w < 200:
            return {'is_valid': False, 'reason': 'Image is too small to be a CT scan. Please upload a proper CT scan image.'}

        # Check 3: Check pixel intensity distribution typical of CT scans
        # CT scans have large dark regions (background) and specific gray patterns
        gray = img_array if len(img_array.shape) == 2 else img_array[:,:,0]
        
        # CT scans typically have >30% very dark pixels (black background)
        dark_pixel_ratio = np.sum(gray < 15) / gray.size
        
        # CT scans have a specific std deviation range (not too uniform, not too noisy)
        std_dev = np.std(gray)
        
        print(f"🔍 CT Validation: grayscale={is_grayscale}, dark_ratio={dark_pixel_ratio:.2f}, std={std_dev:.1f}, size={w}x{h}")
        
        if dark_pixel_ratio > 0.25 and 20 < std_dev < 120:
            # Check 4: OCR check for DICOM metadata as confirmation
            ocr_text = extract_text_with_tesseract(image_bytes)
            if ocr_text:
                text_lower = ocr_text.lower()
                ct_keywords = ['kidney', 'renal', 'abdomen', 'ct', 'kv', 'ma', 'slice', 
                               'window', 'level', 'dicom', 'hounsfield', 'thickness', 'dfov']
                matches = sum(1 for kw in ct_keywords if kw in text_lower)
                print(f"🔍 CT Validation OCR: {matches} keywords found")
                if matches >= 1:
                    return {'is_valid': True, 'reason': f'CT scan confirmed (grayscale + DICOM metadata)'}
            
            # Grayscale with CT-like pixel distribution — accept
            return {'is_valid': True, 'reason': 'CT scan accepted (grayscale medical image)'}
        else:
            return {'is_valid': False, 'reason': 'This does not appear to be a CT scan. Please upload a kidney CT scan image (grayscale, black background).'}

    except Exception as e:
        print(f"⚠️ CT image analysis error: {e}")
        # If we can't even open/analyze the image, reject it
        return {'is_valid': False, 'reason': 'Could not process image. Please upload a valid JPEG or PNG CT scan.'}
  
   
def validate_blood_report(file_bytes, filename):
    """
    Check if uploaded file is a blood report (PDF or image).
    Returns: {'is_valid': bool, 'reason': str, 'extracted_text': str or None}
    """
    extracted_text = None
    filename_lower = filename.lower()

    # Step 1: Extract text
    if filename_lower.endswith('.pdf'):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            extracted_text = extract_text_from_pdf(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    else:
        extracted_text = extract_text_with_tesseract(file_bytes)

    if not extracted_text or len(extracted_text.strip()) < 30:
        # Try HuggingFace OCR as fallback
        if HUGGINGFACE_API_KEY and not filename_lower.endswith('.pdf'):
            extracted_text = extract_text_with_huggingface_ocr(file_bytes)

    if not extracted_text or len(extracted_text.strip()) < 30:
        # If we can't extract any text, use vision model to check
        if openai_client and not filename_lower.endswith('.pdf'):
            try:
                base64_image = base64.b64encode(file_bytes).decode('utf-8')
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Is this image a blood test / laboratory blood report? Reply ONLY: BLOOD_REPORT or NOT_BLOOD_REPORT"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}],
                    max_tokens=10, temperature=0
                )
                ans = response.choices[0].message.content.strip().upper()
                if "BLOOD_REPORT" in ans:
                    return {'is_valid': True, 'reason': 'Blood report confirmed via vision', 'extracted_text': None}
                else:
                    return {'is_valid': False, 'reason': 'This does not appear to be a blood report. Please upload a blood test report (PDF or image).', 'extracted_text': None}
            except Exception as e:
                print(f"⚠️ Vision blood validation error: {e}")
        return {'is_valid': False, 'reason': 'Could not read file content. Please upload a clear blood report image or PDF.', 'extracted_text': None}

    # Step 2: Check extracted text for blood report keywords
    text_lower = extracted_text.lower()
    blood_keywords = [
        'hemoglobin', 'haemoglobin', 'hb', 'wbc', 'rbc', 'platelet', 'leucocyte', 'leukocyte',
        'hematocrit', 'haematocrit', 'mcv', 'mch', 'mchc', 'rdw', 'neutrophil', 'lymphocyte',
        'monocyte', 'eosinophil', 'basophil', 'blood count', 'cbc', 'complete blood',
        'glucose', 'creatinine', 'urea', 'cholesterol', 'triglyceride', 'hba1c',
        'serum', 'plasma', 'blood', 'lab report', 'laboratory', 'pathology',
        'reference range', 'normal range', 'test result', 'g/dl', 'mg/dl', 'mmol',
        '×10', 'x10', 'per µl', 'per ul', 'cells/µl'
    ]
    matches = sum(1 for kw in blood_keywords if kw in text_lower)

    if matches >= 3:
        print(f"✅ Blood report validated: {matches} keywords found")
        return {'is_valid': True, 'reason': f'Blood report confirmed ({matches} medical terms found)', 'extracted_text': extracted_text}
    elif matches >= 1:
        # Partial match — use Groq to double-check
        if groq_client:
            try:
                snippet = extracted_text[:500]
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a document classifier. Reply ONLY with BLOOD_REPORT or NOT_BLOOD_REPORT."},
                        {"role": "user", "content": f"Is this a blood test / lab report?\n\n{snippet}"}
                    ],
                    max_tokens=10, temperature=0
                )
                ans = response.choices[0].message.content.strip().upper()
                if "BLOOD_REPORT" in ans:
                    return {'is_valid': True, 'reason': 'Blood report confirmed by AI', 'extracted_text': extracted_text}
                else:
                    return {'is_valid': False, 'reason': 'This does not appear to be a blood test report. Please upload a blood report.', 'extracted_text': None}
            except:
                pass
        # Default: if we found at least 1 keyword, allow it
        return {'is_valid': True, 'reason': 'Partial blood report match — proceeding', 'extracted_text': extracted_text}
    else:
        return {'is_valid': False, 'reason': 'This does not appear to be a blood report. Please upload a blood test/lab report PDF or image.', 'extracted_text': None}


# Initialize on import
init_groq()
init_openai()
init_huggingface()
init_ollama()