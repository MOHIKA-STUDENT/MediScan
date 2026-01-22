# app.py - Fixed version with better error handling
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify
from flask_cors import CORS
import timm
from torchvision import models
import base64
from io import BytesIO
import warnings
import logging
import json
import traceback
import pdfplumber
import re
from flask import request
from ai_medical_helper import (
    generate_ct_scan_analysis,
    generate_blood_report_analysis,
    analyze_blood_report_pdf,
    analyze_blood_report_image,
    analyze_blood_report_with_disease_models,  # NEW - for disease classification
    extract_text_from_pdf,  # NEW - missing function
    extract_text_with_tesseract,  # NEW - missing function
    init_groq,
    init_openai,
    init_huggingface
)
from dotenv import load_dotenv
load_dotenv()

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj



warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# FIXED: PDF endpoint after app = Flask()
# Add this to your existing app.py (replace the blood report endpoints)


# In app.py - Update the /api/blood-report/analyze-pdf endpoint

@app.route('/api/blood-report/analyze-pdf', methods=['POST'])
def analyze_blood_pdf_with_diseases():
    """Analyze blood report with disease classification"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        all_results = []
        
        for file in files:
            filename = file.filename.lower()
            file_bytes = file.read()
            
            if filename.endswith('.pdf'):
                result = analyze_blood_report_pdf(file_bytes)
                
            elif filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                result = analyze_blood_report_image(file_bytes)
            
            else:
                continue
            
            if result and result.get('success'):
                # Convert NumPy types before adding to results
                all_results.append({
                    'filename': file.filename,
                    'disease_predictions': convert_numpy_types(result.get('disease_predictions')),
                    'num_positive_diseases': int(result.get('num_positive_diseases', 0)),
                    'positive_diseases': result.get('positive_diseases', []),
                    'detected_diseases_summary': result.get('detected_diseases_summary', ''),  # NEW: Quick summary for user
                    'analysis': result.get('analysis'),
                    'ai_provider': result.get('ai_provider'),
                    'has_disease_classification': bool(result.get('has_disease_classification', False))
                })
        
        if not all_results:
            return jsonify({'error': 'No files processed'}), 400
        
        combined_analysis = "\n\n--- FILE SEPARATION ---\n\n".join([
            f"FILE: {r['filename']}\n{r['detected_diseases_summary']}\n\n{r['analysis']}" for r in all_results  # NEW: Include summary upfront
        ])
        
        # Convert all results to ensure JSON serialization
        response_data = {
            'success': True,
            'total_files': len(all_results),
            'detected_diseases_summary': all_results[0].get('detected_diseases_summary', ''),  # NEW: Top-level for frontend
            'ai_analysis': combined_analysis,
            'ai_provider': all_results[0]['ai_provider'],
            'disease_predictions': all_results[0].get('disease_predictions'),
            'num_positive_diseases': int(all_results[0].get('num_positive_diseases', 0)),
            'positive_diseases': all_results[0].get('positive_diseases', []),
            'has_disease_classification': bool(all_results[0].get('has_disease_classification', False)),
            'details': all_results
        }
        
        # Final conversion pass
        return jsonify(convert_numpy_types(response_data))
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/blood-report/analyze-image', methods=['POST'])
def analyze_blood_image():
    """Analyze blood report from image files (alternative endpoint)"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        results = []
        
        for file in files:
            file_bytes = file.read()
            
            result = analyze_blood_report_image(file_bytes)
            
            if result.get('success'):
                results.append({
                    'filename': file.filename,
                    'analysis': result.get('analysis', ''),
                    'ai_provider': result.get('ai_provider', 'Unknown'),
                    'extraction_method': result.get('extraction_method', 'vision')
                })
            else:
                results.append({
                    'filename': file.filename,
                    'error': result.get('error', 'Analysis failed')
                })
        
        successful = [r for r in results if 'error' not in r]
        
        if not successful:
            return jsonify({'error': 'No images analyzed successfully'}), 400
        
        return jsonify({
            'success': True,
            'total_files': len(files),
            'successful_analyses': len(successful),
            'ai_analysis': successful[0]['analysis'] if successful else '',
            'ai_provider': successful[0]['ai_provider'] if successful else 'Unknown',
            'results': results
        })
        
    except Exception as e:
        logging.error(f"Image analysis failed: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


# Import the new functions at the top of app.py
from ai_medical_helper import (
    generate_ct_scan_analysis,
    generate_blood_report_analysis,
    analyze_blood_report_pdf,
    analyze_blood_report_image,
    init_groq,
    init_openai,
    init_huggingface
)

# ----------------------------------
# CONFIGURATION
# ----------------------------------
CLASSES = ['Normal', 'Stone', 'Cyst', 'Tumor']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224  # For pretrained models (EfficientNet, ResNet, DenseNet)
CNN_IMG_SIZE = 128  # For custom CNN models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
MODEL_PATHS = {
    'efficientnet_b0': 'models/trained_models/EfficientNetB0_BEST.pth',
    'resnet50': 'models/trained_models/ResNet50_BEST.pth',
    'densenet121': 'models/trained_models/DenseNet121_BEST.pth',
    # 'simple_cnn': 'models/trained_models/SimpleCNN_scratch.pt',
    'cnn_kidney': 'models/trained_models/best_kidney_model_full.pth'
}

# Conservative confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'efficientnet_b0': 0.75,
    'resnet50': 0.70,
    'densenet121': 0.80,
    # 'simple_cnn': 0.65,
    'cnn_kidney': 0.65
}

# Ensemble weights based on validation performance
ENSEMBLE_WEIGHTS = {
    'densenet121': 0.40,     # 99.60% accuracy
    'efficientnet_b0': 0.35, # 99.25% accuracy  
    'resnet50': 0.25,        # 99.36% accuracy
    'cnn_kidney': 0.15       #100.00% accuracy
}

# ----------------------------------
# PREPROCESSING
# ----------------------------------
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

# Transform for pretrained models (224x224)
val_transform_224 = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ToTensorV2(),
])

# Transform for custom CNN models (128x128)
val_transform_128 = A.Compose([
    A.Resize(CNN_IMG_SIZE, CNN_IMG_SIZE),
    A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ToTensorV2(),
])

# Conservative TTA transforms
def get_tta_transforms(img_size):
    return [
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=1.0),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]),
    ]

def preprocess_image_simple(image_data):
    """Simple preprocessing"""
    try:
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_data))
        else:
            image = Image.open(image_data)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_np = np.array(image)
        
        # Handle grayscale medical images
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np[:,:,0], cv2.COLOR_GRAY2RGB)
        
        return image_np
    except Exception as e:
        logging.error(f"Image preprocessing error: {e}")
        logging.error(traceback.format_exc())
        return None

# ----------------------------------
# MODEL ARCHITECTURES
# ----------------------------------
class KidneyCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(KidneyCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*(CNN_IMG_SIZE//16)*(CNN_IMG_SIZE//16), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),

#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)

def create_efficientnet_b0():
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.28),
        nn.Linear(256, NUM_CLASSES)
    )
    return model

def create_resnet50():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model

def create_densenet121():
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model

# ----------------------------------
# MODEL LOADING
# ----------------------------------
loaded_models = {}
torch.serialization.add_safe_globals([KidneyCNN])

def load_model(model_name):
    """Load and cache models"""
    if model_name in loaded_models:
        return loaded_models[model_name]

    try:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path:
            logging.error(f"Unknown model: {model_name}")
            return None
            
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            return None

        # Create model architecture
        if model_name == 'efficientnet_b0':
            model = create_efficientnet_b0()
        elif model_name == 'resnet50':
            model = create_resnet50()
        elif model_name == 'densenet121':
            model = create_densenet121()
        # elif model_name == 'simple_cnn':
        #     model = SimpleCNN(NUM_CLASSES)
        elif model_name == 'cnn_kidney':
            model = KidneyCNN(num_classes=NUM_CLASSES)
        else:
            logging.error(f"Unknown model architecture: {model_name}")
            return None

        # Load weights
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        # Handle different save formats
        if isinstance(checkpoint, dict):
            # Checkpoint saved as dictionary with 'model_state_dict' key
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            # Checkpoint saved as raw state_dict
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            # Checkpoint is just a state_dict
            else:
                model.load_state_dict(checkpoint)
        elif isinstance(checkpoint, nn.Module):
            # Checkpoint saved as entire model - extract state_dict
            model.load_state_dict(checkpoint.state_dict())
        else:
            # Assume it's a state_dict
            model.load_state_dict(checkpoint)

        model.to(DEVICE).eval()
        loaded_models[model_name] = model
        logging.info(f"✓ {model_name} loaded successfully")
        return model

    except Exception as e:
        logging.error(f"Error loading {model_name}: {str(e)}")
        logging.error(traceback.format_exc())
        return None

# ----------------------------------
# PREDICTION FUNCTIONS
# ----------------------------------
def get_transform_for_model(model_name):
    """Get appropriate transform based on model"""
    if model_name in [ 'cnn_kidney']:
        return val_transform_128
    return val_transform_224

def predict_single_safe(model, image_np, model_name):
    """Single prediction with exact training preprocessing"""
    try:
        transform = get_transform_for_model(model_name)
        processed = transform(image=image_np)
        image_tensor = processed['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

            return {
                'predicted_class': CLASSES[predicted_class],
                'predicted_index': predicted_class,
                'confidence': float(confidence),
                'reliable': confidence > CONFIDENCE_THRESHOLDS.get(model_name, 0.70),
                'all_probabilities': {CLASSES[i]: float(prob) for i, prob in enumerate(probs[0])}
            }
            
    except Exception as e:
        logging.error(f"Prediction failed for {model_name}: {e}")
        logging.error(traceback.format_exc())
        return None

def predict_with_safe_tta(model, image_np, model_name):
    """Very conservative TTA"""
    try:
        img_size = CNN_IMG_SIZE if model_name in ['cnn_kidney'] else IMG_SIZE
        tta_transforms = get_tta_transforms(img_size)
        predictions = []
        
        for transform in tta_transforms:
            processed = transform(image=image_np)
            image_tensor = processed['image'].unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(image_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Average predictions
        avg_probs = torch.stack(predictions).mean(dim=0)
        predicted_class = torch.argmax(avg_probs, dim=1).item()
        confidence = avg_probs[0][predicted_class].item()
        
        # Calculate TTA agreement
        individual_preds = [torch.argmax(pred, dim=1).item() for pred in predictions]
        tta_agreement = individual_preds.count(predicted_class) / len(individual_preds)
        
        return {
            'predicted_class': CLASSES[predicted_class],
            'predicted_index': predicted_class,
            'confidence': float(confidence),
            'tta_agreement': float(tta_agreement),
            'reliable': confidence > CONFIDENCE_THRESHOLDS.get(model_name, 0.70) and tta_agreement >= 0.67,
            'all_probabilities': {CLASSES[i]: float(prob) for i, prob in enumerate(avg_probs[0])}
        }
        
    except Exception as e:
        logging.error(f"TTA prediction failed for {model_name}: {e}")
        logging.error(traceback.format_exc())
        return None

def smart_ensemble_predict(image_np, use_tta=False):
    """Smart ensemble using only the top 3 models"""
    try:
        model_predictions = []
        model_names = ['densenet121', 'efficientnet_b0', 'resnet50', 'cnn_kidney']
        weights = [ENSEMBLE_WEIGHTS[name] for name in model_names]
        
        for model_name in model_names:
            model = load_model(model_name)
            if model is None:
                logging.warning(f"Skipping {model_name} - model not available")
                continue
                
            if use_tta:
                pred = predict_with_safe_tta(model, image_np, model_name)
            else:
                pred = predict_single_safe(model, image_np, model_name)
            
            if pred is not None:
                model_predictions.append(pred)
            else:
                idx = model_names.index(model_name)
                weights[idx] = 0
        
        if len(model_predictions) < 2:
            logging.error("Not enough models available for ensemble")
            return None
        
        # Normalize weights
        total_weight = sum(w for w, pred in zip(weights, model_predictions) if pred is not None)
        if total_weight == 0:
            return None
            
        normalized_weights = [w / total_weight for w in weights if w > 0]
        
        # Weighted ensemble
        ensemble_probs = {}
        for class_name in CLASSES:
            weighted_sum = 0
            for pred, weight in zip(model_predictions, normalized_weights):
                weighted_sum += pred['all_probabilities'][class_name] * weight
            ensemble_probs[class_name] = weighted_sum
        
        predicted_class = max(ensemble_probs, key=ensemble_probs.get)
        predicted_idx = CLASSES.index(predicted_class)
        confidence = ensemble_probs[predicted_class]
        
        # Check model agreement
        individual_preds = [pred['predicted_class'] for pred in model_predictions]
        agreement_score = individual_preds.count(predicted_class) / len(individual_preds)
        
        return {
            'predicted_class': predicted_class,
            'predicted_index': predicted_idx,
            'confidence': float(confidence),
            'model_agreement': float(agreement_score),
            'reliable': confidence > 0.75 and agreement_score >= 0.67,
            'ensemble_size': len(model_predictions),
            'all_probabilities': ensemble_probs,
            'individual_predictions': {name: pred for name, pred in zip(model_names, model_predictions) if pred is not None}
        }
        
    except Exception as e:
        logging.error(f"Ensemble prediction failed: {e}")
        logging.error(traceback.format_exc())
        return None

# ----------------------------------
# ROUTES
# ----------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    available_models = [m for m in MODEL_PATHS.keys() if os.path.exists(MODEL_PATHS[m])]
    return jsonify({
        'status': 'healthy', 
        'device': str(DEVICE),
        'available_models': available_models,
        'total_models': len(MODEL_PATHS),
        'models_found': len(available_models)
    })





@app.route('/models', methods=['GET'])
def get_models():
    available = [m for m in MODEL_PATHS.keys() if os.path.exists(MODEL_PATHS[m])]
    missing = [m for m in MODEL_PATHS.keys() if not os.path.exists(MODEL_PATHS[m])]
    return jsonify({
        'available_models': available,
        'missing_models': missing,
        'confidence_thresholds': CONFIDENCE_THRESHOLDS,
        'ensemble_weights': ENSEMBLE_WEIGHTS
    })

# Alternative route names (for backward compatibility)
@app.route('/analyze_safe', methods=['POST'])
def analyze_safe():
    """Alias for /analyze"""
    return analyze()

@app.route('/analyze_ensemble_safe', methods=['POST'])
def analyze_ensemble_safe():
    """Alias for /ensemble_analyze"""
    return ensemble_analyze()

# Main routes - these are called by the frontend
@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analyze route - same as analyze_safe"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        model_name = request.form.get('model', 'densenet121')
        use_tta = request.form.get('use_tta', 'false').lower() == 'true'

        logging.info(f"Analyzing {len(files)} files with model: {model_name}, TTA: {use_tta}")

        model = load_model(model_name)
        if model is None:
            return jsonify({'error': f'Model {model_name} not available or failed to load'}), 500

        results = []
        for file in files:
            if file.filename == '':
                continue

            logging.info(f"Processing file: {file.filename}")
            image_np = preprocess_image_simple(file.stream)
            if image_np is None:
                results.append({'filename': file.filename, 'error': 'Image processing failed'})
                continue

            if use_tta:
                prediction = predict_with_safe_tta(model, image_np, model_name)
            else:
                prediction = predict_single_safe(model, image_np, model_name)
                
            if prediction:
                prediction['filename'] = file.filename
                prediction['model_used'] = model_name
                prediction['tta_enabled'] = use_tta
                results.append(prediction)
                logging.info(f"✓ {file.filename}: {prediction['predicted_class']} ({prediction['confidence']:.2%})")
            else:
                results.append({'filename': file.filename, 'error': 'Prediction failed'})

        return jsonify({
            'model_used': model_name,
            'tta_enabled': use_tta,
            'results': results,
            'total_files': len(files),
            'successful_predictions': len([r for r in results if 'error' not in r])
        })

    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/ensemble_analyze', methods=['POST'])
def ensemble_analyze():
    """Main ensemble route - same as analyze_ensemble_safe"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        use_tta = request.form.get('use_tta', 'false').lower() == 'true'

        logging.info(f"Ensemble analyzing {len(files)} files, TTA: {use_tta}")

        results = []
        for file in files:
            if file.filename == '':
                continue

            logging.info(f"Processing file: {file.filename}")
            image_np = preprocess_image_simple(file.stream)
            if image_np is None:
                results.append({'filename': file.filename, 'error': 'Image processing failed'})
                continue

            prediction = smart_ensemble_predict(image_np, use_tta)
            if prediction:
                prediction['filename'] = file.filename
                prediction['tta_enabled'] = use_tta
                results.append(prediction)
                logging.info(f"✓ {file.filename}: {prediction['predicted_class']} ({prediction['confidence']:.2%})")
            else:
                results.append({'filename': file.filename, 'error': 'Ensemble prediction failed'})

        return jsonify({
            'method': 'smart_ensemble',
            'tta_enabled': use_tta,
            'results': results,
            'total_files': len(files),
            'successful_predictions': len([r for r in results if 'error' not in r])
        })

    except Exception as e:
        logging.error(f"Ensemble analysis failed: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Ensemble analysis failed: {str(e)}'}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Batch analysis with safe preprocessing"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        results = []

        for file in files:
            if file.filename == '':
                continue

            image_np = preprocess_image_simple(file.stream)
            if image_np is None:
                results.append({'filename': file.filename, 'error': 'Image processing failed'})
                continue

            file_results = {}
            for model_name in MODEL_PATHS.keys():
                model = load_model(model_name)
                if model:
                    prediction = predict_single_safe(model, image_np, model_name)
                    file_results[model_name] = prediction
                else:
                    file_results[model_name] = {'error': 'Model not available'}

            results.append({
                'filename': file.filename,
                'model_predictions': file_results
            })

        return jsonify({
            'method': 'batch_safe',
            'results': results,
            'total_files': len(files)
        })

    except Exception as e:
        logging.error(f"Batch analysis failed: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500
    
    
    
# ========================================
# AI-ENHANCED CT SCAN ANALYSIS ROUTE
# ========================================

@app.route('/ensemble_analyze_with_ai', methods=['POST'])
def ensemble_analyze_with_ai():
    """
    CT Scan Analysis with AI Medical Report
    Combines ML prediction + AI medical guidance
    """
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        use_tta = request.form.get('use_tta', 'false').lower() == 'true'

        logging.info(f"AI-Enhanced Ensemble analyzing {len(files)} files, TTA: {use_tta}")

        results = []
        for file in files:
            if file.filename == '':
                continue

            logging.info(f"Processing file: {file.filename}")
            
            # Step 1: ML Prediction (your existing ensemble code)
            image_np = preprocess_image_simple(file.stream)
            if image_np is None:
                results.append({'filename': file.filename, 'error': 'Image processing failed'})
                continue

            prediction = smart_ensemble_predict(image_np, use_tta)
            
            if prediction:
                # Step 2: AI Medical Analysis
                try:
                    ai_report = generate_ct_scan_analysis(prediction)
                    
                    result = {
                        'filename': file.filename,
                        'tta_enabled': use_tta,
                        
                        # ML Results
                        'predicted_class': prediction['predicted_class'],
                        'confidence': float(prediction['confidence']),
                        'model_agreement': float(prediction.get('model_agreement', 0)),
                        'ensemble_size': prediction.get('ensemble_size', 0),
                        'all_probabilities': prediction['all_probabilities'],
                        
                        # AI Medical Report
                        'ai_analysis': ai_report.get('analysis', ''),
                        'ai_provider': ai_report.get('ai_provider', 'Unknown'),
                        'has_ai_report': ai_report.get('success', False)
                    }
                    
                    results.append(result)
                    logging.info(f"✅ {file.filename}: {prediction['predicted_class']} ({prediction['confidence']:.2%}) + AI Report")
                    
                except Exception as ai_error:
                    logging.error(f"AI analysis failed for {file.filename}: {ai_error}")
                    # Still return ML results even if AI fails
                    result = {
                        'filename': file.filename,
                        'predicted_class': prediction['predicted_class'],
                        'confidence': float(prediction['confidence']),
                        'all_probabilities': prediction['all_probabilities'],
                        'ai_analysis': 'AI analysis temporarily unavailable',
                        'has_ai_report': False
                    }
                    results.append(result)
            else:
                results.append({'filename': file.filename, 'error': 'Ensemble prediction failed'})

        return jsonify({
            'method': 'ai_enhanced_ensemble',
            'tta_enabled': use_tta,
            'results': results,
            'total_files': len(files),
            'successful_predictions': len([r for r in results if 'error' not in r])
        })

    except Exception as e:
        logging.error(f"AI-Enhanced analysis failed: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


# ========================================
# BLOOD REPORT AI ANALYSIS ROUTE
# ========================================

@app.route('/api/blood-report/analyze', methods=['POST'])
def analyze_blood_report():
    """
    Blood Report AI Analysis
    Accepts blood test parameters and returns AI-generated insights
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No blood data provided'}), 400
        
        # Expected blood parameters (example)
        blood_params = {
            'hemoglobin': data.get('hemoglobin'),
            'wbc': data.get('wbc'),
            'rbc': data.get('rbc'),
            'platelets': data.get('platelets'),
            'glucose': data.get('glucose'),
            'creatinine': data.get('creatinine'),
            'urea': data.get('urea'),
            'sodium': data.get('sodium'),
            'potassium': data.get('potassium'),
            'calcium': data.get('calcium'),
        }
        
        # Remove None values
        blood_params = {k: v for k, v in blood_params.items() if v is not None}
        
        if not blood_params:
            return jsonify({'error': 'No valid blood parameters provided'}), 400
        
        # Generate AI analysis
        ai_report = generate_blood_report_analysis(blood_params)
        
        return jsonify({
            'success': True,
            'blood_data': blood_params,
            'ai_analysis': ai_report.get('analysis', ''),
            'ai_provider': ai_report.get('ai_provider', 'Unknown'),
            'has_analysis': ai_report.get('success', False)
        })
        
    except Exception as e:
        logging.error(f"Blood report analysis failed: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


# ========================================
# HEALTH CHECK WITH AI STATUS
# ========================================

@app.route('/health', methods=['GET'])
def health_check_with_ai():
    """Health check including AI services status"""
    from ai_medical_helper import groq_client, openai_client
    
    available_models = [m for m in MODEL_PATHS.keys() if os.path.exists(MODEL_PATHS[m])]
    
    return jsonify({
        'status': 'healthy', 
        'device': str(DEVICE),
        'available_models': available_models,
        'total_models': len(MODEL_PATHS),
        'models_found': len(available_models),
        'ai_services': {
            'groq': groq_client is not None,
            'openai': openai_client is not None,
            'status': 'AI services ready' if groq_client or openai_client else 'Using fallback templates'
        }
    })


if __name__ == '__main__':
    logging.info("=" * 60)
    logging.info("Starting CT Scan Analysis Server with AI")
    logging.info("=" * 60)
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Image sizes: Pretrained={IMG_SIZE}, Custom CNN={CNN_IMG_SIZE}")
    
    # Initialize AI services
    logging.info("\n🤖 Initializing AI Services...")
    groq_available = init_groq()
    openai_available = init_openai()
    
    if groq_available:
        logging.info("✅ Groq AI ready for medical analysis")
    elif openai_available:
        logging.info("✅ OpenAI ready for medical analysis")
    else:
        logging.warning("⚠️ No AI services available - using template responses")
    
    # Check model availability
    available = []
    missing = []
    for model_name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            available.append(model_name)
            logging.info(f"✓ {model_name}: {path}")
        else:
            missing.append(model_name)
            logging.warning(f"✗ {model_name}: NOT FOUND at {path}")
    
    logging.info("=" * 60)
    logging.info(f"Available ML models: {len(available)}/{len(MODEL_PATHS)}")
    if missing:
        logging.warning(f"Missing models: {', '.join(missing)}")
    logging.info("=" * 60)
    logging.info("\n🚀 Server ready at http://0.0.0.0:5000")
    logging.info("📋 New endpoints:")
    logging.info("   POST /ensemble_analyze_with_ai - CT Scan with AI report")
    logging.info("   POST /api/blood-report/analyze - Blood report analysis")
    logging.info("=" * 60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)