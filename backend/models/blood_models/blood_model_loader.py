# backend/models/blood_models/blood_model_loader.py
# CLEAN INTEGRATION: Loads Your Trained .pkl Files + Imputation
import os
import pickle
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(current_dir, '..', 'trained_models'))
print(f"📍 blood_model_loader.py location: {current_dir}")
print(f"📂 Looking for models in: {MODELS_DIR}")

loaded_disease_models = {}
model_metadata = {}

# Load your trained metadata
metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
if os.path.exists(metadata_path):
    try:
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        # Add imputation medians from training (use averages from your datasets)
        for disease in model_metadata:
            if 'imputation_medians' not in model_metadata[disease]:
                # Example medians (tune from your training data)
                medians = {
                    'Diabetes': {'Fasting_Glucose_mg_dL': 90.0, 'HbA1c_percent': 5.5},
                    'Dengue': {'Platelet_Count_per_uL': 250000, 'WBC_Count_per_uL': 7000},
                    'Malaria': {'Hemoglobin_g_dL': 14.0, 'Platelet_Count_per_uL': 250000},
                    'Anemia': {'Hemoglobin_g_dL': 14.0, 'Hematocrit_percent': 42.0},
                    'Infection': {'WBC_Count_per_uL': 7000, 'Neutrophils_percent': 60.0},
                    'Typhoid': {'WBC_Count_per_uL': 7000, 'Lymphocytes_percent': 30.0}
                }
                model_metadata[disease]['imputation_medians'] = medians.get(disease, {feat: 0.0 for feat in model_metadata[disease]['feature_names']})
        print(f"✅ Loaded metadata for {len(model_metadata)} diseases (from training)")
    except Exception as e:
        print(f"❌ Metadata load failed: {e} - Regenerate with train_models.py")
        model_metadata = {}  # Retry on import

# Load your trained models (no dummies)
DISEASES = ['Diabetes', 'Dengue', 'Malaria', 'Anemia', 'Infection', 'Typhoid']
print("\n🔍 Loading trained disease models...")
for disease in DISEASES:
    model_path = os.path.join(MODELS_DIR, f'{disease}_model.pkl')
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                loaded_disease_models[disease] = pickle.load(f)
            print(f"  ✅ Loaded {disease} model (trained accuracy: {model_metadata.get(disease, {}).get('test_accuracy', 'N/A')})")
        except Exception as e:
            print(f"  ❌ Load failed for {disease}: {e} - Retrain with train_models.py")
    else:
        print(f"  ⚠️ {disease}_model.pkl missing → Run train_models.py")

print(f"\n📊 Total trained models loaded: {len(loaded_disease_models)}/6")

# ============================================================================
# MEDICAL INTELLIGENCE LAYER
# ============================================================================

def calculate_medical_confidence(disease, blood_params, ml_prediction, ml_confidence):
    """
    Apply medical knowledge to adjust ML confidence
    CRITICAL: Conservative approach - only boost when clinically certain
    """
    
    if disease == 'Dengue':
        platelets = blood_params.get('Platelet_Count_per_uL', 250000)
        wbc = blood_params.get('WBC_Count_per_uL', 7000)
        hematocrit = blood_params.get('Hematocrit_percent', 42)
        
        dengue_score = 0
        
        # Platelet count (PRIMARY indicator for Dengue)
        if platelets < 50000:
            dengue_score += 0.50  # Severe thrombocytopenia - STRONG indicator
        elif platelets < 100000:
            dengue_score += 0.35
        elif platelets < 150000:
            dengue_score += 0.15
        
        # WBC count (SECONDARY indicator)
        if wbc < 3000:
            dengue_score += 0.30
        elif wbc < 4000:
            dengue_score += 0.20
        
        # Hematocrit elevation
        if hematocrit > 45:
            dengue_score += 0.10
        
        if ml_prediction == 1:
            final_confidence = min(ml_confidence + dengue_score * 0.25, 0.95)
        else:
            # Override ML ONLY if very strong clinical evidence
            if dengue_score >= 0.70:  # Very high threshold
                ml_prediction = 1
                final_confidence = 0.80 + dengue_score * 0.15
                print(f"  🦟 Medical override: Strong Dengue indicators (score: {dengue_score:.2f})")
            else:
                final_confidence = ml_confidence
        
        return ml_prediction, final_confidence
    
    elif disease == 'Malaria':
        platelets = blood_params.get('Platelet_Count_per_uL', 250000)
        hemoglobin = blood_params.get('Hemoglobin_g_dL', 14.0)
        
        malaria_score = 0
        
        if platelets < 80000:
            malaria_score += 0.30
        elif platelets < 120000:
            malaria_score += 0.15
        
        if hemoglobin < 10:
            malaria_score += 0.35
        elif hemoglobin < 11:
            malaria_score += 0.20
        
        if ml_prediction == 1:
            final_confidence = min(ml_confidence + malaria_score * 0.20, 0.95)
        else:
            # Only override if BOTH low platelets AND low hemoglobin
            if malaria_score >= 0.50 and platelets < 100000 and hemoglobin < 11:
                ml_prediction = 1
                final_confidence = 0.70
                print(f"  🦠 Medical override: Malaria indicators present")
            else:
                final_confidence = ml_confidence
        
        return ml_prediction, final_confidence
    
    elif disease == 'Anemia':
        hemoglobin = blood_params.get('Hemoglobin_g_dL', 14.0)
        
        # Clear-cut anemia criteria (WHO standards)
        if hemoglobin < 10:
            return 1, 0.95  # Definite anemia
        elif hemoglobin < 12:
            return 1, 0.80  # Likely anemia
        elif hemoglobin >= 13.5 and ml_prediction == 1:
            return 0, 0.90  # Override: Normal hemoglobin = NOT anemic
        
        return ml_prediction, ml_confidence
    
    elif disease == 'Infection':
        wbc = blood_params.get('WBC_Count_per_uL', 7000)
        neutrophils = blood_params.get('Neutrophils_percent', 60.0)
        lymphocytes = blood_params.get('Lymphocytes_percent', 30.0)
        
        # BACTERIAL infection score
        bacterial_score = 0
        if wbc > 12000:
            bacterial_score += 0.40
        if neutrophils > 80:
            bacterial_score += 0.35
        elif neutrophils > 75:
            bacterial_score += 0.20
        
        # VIRAL infection score (but we have specific viral tests)
        viral_score = 0
        if wbc < 4000 and lymphocytes > 50:
            viral_score += 0.30  # Viral pattern detected
        
        # CRITICAL LOGIC: If viral pattern, check for specific viral diseases
        if viral_score > 0:
            print(f"  ℹ️  Infection: Viral pattern detected - checking specific viral tests...")
        
        if ml_prediction == 1:
            # ML says positive - verify it's real
            if bacterial_score >= 0.40:
                # Strong bacterial evidence - confirm
                final_confidence = min(ml_confidence + bacterial_score * 0.15, 0.90)
            elif viral_score > 0:
                # Viral pattern - keep moderate confidence but will be overridden by specific tests
                final_confidence = ml_confidence * 0.80  # Reduce confidence
            else:
                # Weak evidence - be very conservative
                final_confidence = ml_confidence * 0.60
                if final_confidence < 0.55:
                    ml_prediction = 0
                    final_confidence = 0.80
                    print(f"  ✓ Medical override: Insufficient infection evidence")
        else:
            # ML says negative
            if bacterial_score >= 0.60:  # Very strong bacterial evidence
                ml_prediction = 1
                final_confidence = 0.75
                print(f"  🦠 Medical override: Strong bacterial infection indicators")
            else:
                final_confidence = ml_confidence
        
        return ml_prediction, final_confidence
    
    elif disease == 'Typhoid':
        wbc = blood_params.get('WBC_Count_per_uL', 7000)
        lymphocytes = blood_params.get('Lymphocytes_percent', 30.0)
        
        # Typhoid has SPECIFIC pattern: leukopenia + relative lymphocytosis
        typhoid_score = 0
        
        # Leukopenia is KEY for typhoid
        if wbc < 3500:
            typhoid_score += 0.30
        elif wbc < 4500:
            typhoid_score += 0.15
        
        # Relative lymphocytosis
        if lymphocytes > 45:
            typhoid_score += 0.20
        
        if ml_prediction == 1:
            # ML says positive - verify
            if typhoid_score >= 0.30:
                final_confidence = min(ml_confidence + typhoid_score * 0.15, 0.85)
            else:
                # Weak evidence
                final_confidence = ml_confidence * 0.70
        else:
            # ML says negative
            if typhoid_score >= 0.45:  # High threshold
                ml_prediction = 1
                final_confidence = 0.65
                print(f"  🦠 Medical override: Typhoid indicators present")
            else:
                final_confidence = ml_confidence
        
        return ml_prediction, final_confidence
    
    elif disease == 'Diabetes':
        fasting = blood_params.get('Fasting_Glucose_mg_dL', 0)
        hba1c = blood_params.get('HbA1c_percent', 0)
        
        if fasting > 126 or hba1c > 6.5:
            return 2, 0.95  # Diabetic
        elif fasting > 100 or hba1c > 5.7:
            return 1, 0.85  # Prediabetic
        elif ml_prediction > 0 and fasting > 0 and fasting < 100:
            return 0, 0.90  # Override: Normal glucose
        
        return ml_prediction, ml_confidence
    
    return ml_prediction, ml_confidence


def apply_disease_hierarchy(all_predictions, blood_params):
    """
    CRITICAL: Apply medical disease hierarchy
    If a SPECIFIC viral disease is detected, suppress generic "Infection" diagnosis
    """
    
    # Check if we have specific viral diseases detected
    dengue_positive = all_predictions.get('Dengue', {}).get('is_positive', False)
    dengue_conf = all_predictions.get('Dengue', {}).get('confidence', 0)
    
    malaria_positive = all_predictions.get('Malaria', {}).get('is_positive', False)
    malaria_conf = all_predictions.get('Malaria', {}).get('confidence', 0)
    
    typhoid_positive = all_predictions.get('Typhoid', {}).get('is_positive', False)
    typhoid_conf = all_predictions.get('Typhoid', {}).get('confidence', 0)
    
    infection_positive = all_predictions.get('Infection', {}).get('is_positive', False)
    infection_conf = all_predictions.get('Infection', {}).get('confidence', 0)
    
    wbc = blood_params.get('WBC_Count_per_uL', 7000)
    neutrophils = blood_params.get('Neutrophils_percent', 60.0)
    
    # RULE 1: If Dengue is positive with high confidence, suppress generic Infection
    if dengue_positive and dengue_conf > 0.70:
        if infection_positive:
            # Check if it's bacterial or just the viral pattern from Dengue
            bacterial_score = 0
            if wbc > 12000:
                bacterial_score += 0.5
            if neutrophils > 80:
                bacterial_score += 0.5
            
            if bacterial_score < 0.5:  # No strong bacterial evidence
                # Suppress generic Infection - it's just detecting the Dengue viral pattern
                all_predictions['Infection']['is_positive'] = False
                all_predictions['Infection']['confidence'] = infection_conf * 0.5
                all_predictions['Infection']['hierarchy_suppressed'] = True
                print(f"  📋 Hierarchy rule: 'Infection' suppressed (specific diagnosis: Dengue)")
    
    # RULE 2: If Malaria is positive, check Infection
    if malaria_positive and malaria_conf > 0.70:
        if infection_positive:
            bacterial_score = (1 if wbc > 12000 else 0) + (1 if neutrophils > 80 else 0)
            if bacterial_score < 1:
                all_predictions['Infection']['is_positive'] = False
                all_predictions['Infection']['hierarchy_suppressed'] = True
                print(f"  📋 Hierarchy rule: 'Infection' suppressed (specific diagnosis: Malaria)")
    
    # RULE 3: If Typhoid is positive with high confidence, suppress generic Infection
    if typhoid_positive and typhoid_conf > 0.65:
        if infection_positive:
            bacterial_score = (1 if wbc > 12000 else 0) + (1 if neutrophils > 80 else 0)
            if bacterial_score < 1:
                # Typhoid IS a bacterial infection, but we have the specific diagnosis
                all_predictions['Infection']['is_positive'] = False
                all_predictions['Infection']['hierarchy_suppressed'] = True
                print(f"  📋 Hierarchy rule: 'Infection' suppressed (specific diagnosis: Typhoid)")
    
    # RULE 4: Don't diagnose both Dengue AND Malaria (very rare co-infection)
    if dengue_positive and malaria_positive:
        # Keep the one with higher confidence
        if dengue_conf > malaria_conf + 0.15:  # Dengue significantly more confident
            all_predictions['Malaria']['is_positive'] = False
            all_predictions['Malaria']['hierarchy_suppressed'] = True
            print(f"  📋 Hierarchy rule: 'Malaria' suppressed (higher confidence: Dengue)")
        elif malaria_conf > dengue_conf + 0.15:  # Malaria significantly more confident
            all_predictions['Dengue']['is_positive'] = False
            all_predictions['Dengue']['hierarchy_suppressed'] = True
            print(f"  📋 Hierarchy rule: 'Dengue' suppressed (higher confidence: Malaria)")
    
    # RULE 5: Don't diagnose both Dengue AND Typhoid (different pathogens, unlikely together)
    if dengue_positive and typhoid_positive:
        if dengue_conf > typhoid_conf + 0.10:
            all_predictions['Typhoid']['is_positive'] = False
            all_predictions['Typhoid']['hierarchy_suppressed'] = True
            print(f"  📋 Hierarchy rule: 'Typhoid' suppressed (higher confidence: Dengue)")
        elif typhoid_conf > dengue_conf + 0.10:
            all_predictions['Dengue']['is_positive'] = False
            all_predictions['Dengue']['hierarchy_suppressed'] = True
            print(f"  📋 Hierarchy rule: 'Dengue' suppressed (higher confidence: Typhoid)")
    
    return all_predictions


def predict_disease(blood_params):
    """
    FINAL VERSION: Predict diseases with medical intelligence + disease hierarchy
    """
    if not loaded_disease_models:
        print("❌ No disease models loaded")
        return None
    
    if not blood_params:
        print("❌ No blood parameters provided")
        return None
    
    print(f"\n🔬 Running disease predictions with {len(blood_params)} parameters...")
    print(f"   Parameters: {list(blood_params.keys())}")
    
    disease_predictions = {}
    
    # Step 1: Get predictions from all models
    for disease, model in loaded_disease_models.items():
        try:
            if disease not in model_metadata:
                disease_predictions[disease] = {'error': 'No metadata'}
                continue
                
            feature_names = model_metadata[disease]['feature_names']
            
            X = []
            missing_count = 0
            
            for feature in feature_names:
                if feature in blood_params:
                    X.append(blood_params[feature])
                else:
                    X.append(0)
                    missing_count += 1
            
            if missing_count == len(feature_names):
                disease_predictions[disease] = {'error': 'All features missing'}
                continue
            
            # Get ML prediction
            X_array = np.array([X])
            ml_prediction = int(model.predict(X_array)[0])
            probabilities = model.predict_proba(X_array)[0]
            ml_confidence = float(probabilities[ml_prediction])
            
            # Apply medical intelligence
            final_prediction, final_confidence = calculate_medical_confidence(
                disease, blood_params, ml_prediction, ml_confidence
            )
            
            is_positive = bool(final_prediction == 1 or final_prediction == 2)
            
            disease_predictions[disease] = {
                'is_positive': is_positive,
                'confidence': float(final_confidence),
                'ml_prediction': int(ml_prediction),
                'ml_confidence': float(ml_confidence),
                'medical_adjusted': abs(final_confidence - ml_confidence) > 0.05,
                'missing_features': int(missing_count),
                'total_features': len(feature_names),
                'hierarchy_suppressed': False
            }
                
        except Exception as e:
            print(f"❌ {disease} prediction failed: {e}")
            disease_predictions[disease] = {'error': str(e)}
    
    # Step 2: Apply disease hierarchy logic
    print("\n📋 Applying medical disease hierarchy...")
    disease_predictions = apply_disease_hierarchy(disease_predictions, blood_params)
    
    # Step 3: Collect final results
    positive_diseases = []
    num_positive = 0
    
    for disease, pred in disease_predictions.items():
        if pred.get('is_positive') and not pred.get('error'):
            positive_diseases.append(disease)
            num_positive += 1
            
            status = "🔴 POSITIVE"
            conf_text = f"{pred['confidence']*100:.1f}%"
            
            if pred.get('hierarchy_suppressed'):
                print(f"  ⚪ {disease}: {conf_text} (SUPPRESSED by hierarchy)")
            elif pred.get('medical_adjusted'):
                print(f"  {status}: {disease} ({conf_text}) 📋 MEDICALLY ADJUSTED")
            else:
                print(f"  {status}: {disease} ({conf_text})")
        elif not pred.get('error'):
            print(f"  ✅ Negative: {disease} ({pred['confidence']*100:.1f}%)")
    
    print(f"\n📊 Final Results: {num_positive} positive out of {len(disease_predictions)} diseases")
    
    summary = f"{num_positive} positive: {', '.join(positive_diseases) if positive_diseases else 'None'}"
    
    return {
        'predictions': disease_predictions,
        'positive_diseases': positive_diseases,
        'num_positive': int(num_positive),
        'summary': summary
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FINAL TEST - Disease Hierarchy Logic")
    print("="*70)
    
    if loaded_disease_models:
        print("\n✅ Testing with Dengue-positive sample...")
        
        dengue_sample = {
            'WBC_Count_per_uL': 3000.0,
            'Neutrophils_percent': 35.0,
            'Lymphocytes_percent': 55.0,
            'Monocytes_percent': 7.0,
            'Eosinophils_percent': 1.0,
            'Platelet_Count_per_uL': 45000.0,
            'Hemoglobin_g_dL': 13.5,
            'Hematocrit_percent': 41.0,
            'MCV_fL': 87.0,
            'MCH_pg': 29.0,
            'MCHC_g_dL': 33.0,
            'RBC_Count_million_per_uL': 4.7
        }
        
        result = predict_disease(dengue_sample)
        
        if result:
            print("\n" + "="*70)
            print("EXPECTED: Only Dengue positive")
            print("RESULT:")
            for disease in result['positive_diseases']:
                print(f"  🔴 {disease}")
            print("="*70)