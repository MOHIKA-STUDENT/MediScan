import os
import torch
import torch.nn as nn
import timm
from torchvision import models
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

class ModelLoader:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['Normal', 'Stone', 'Cyst', 'Tumor']
        self.num_classes = len(self.classes)
        self.models = {}
        self.transform = self._get_transform()
        
    def _get_transform(self):
        """Get the same preprocessing transform used during training"""
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ])
    
    def _create_efficientnet_b0(self):
        """Create EfficientNetB0 model with same architecture as training"""
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=self.num_classes)
        
        # Recreate the custom classifier head used during training
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35 * 0.8),
            nn.Linear(256, self.num_classes)
        )
        
        return model
    
    def _create_resnet50(self):
        """Create ResNet50 model with same architecture as training"""
        model = models.resnet50(pretrained=False)
        
        # Recreate the custom classifier head
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes)
        )
        
        return model
    
    def _create_densenet121(self):
        """Create DenseNet121 model with same architecture as training"""
        model = models.densenet121(pretrained=False)
        
        # Recreate the custom classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier.in_features, 512),
            nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes)
        )
        
        return model
    
    
    
    
    def load_model(self, model_name):
        """Load a specific model"""
        model_path = self.models_dir / f"{model_name}_BEST.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model based on name
        if model_name == "EfficientNetB0":
            model = self._create_efficientnet_b0()
        elif model_name == "ResNet50":
            model = self._create_resnet50()
        elif model_name == "DenseNet121":
            model = self._create_densenet121()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Load the saved state
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode and move to device
        model.eval()
        model.to(self.device)
        
        self.models[model_name] = model
        print(f"Loaded {model_name} successfully")
        
        return model
    
    def load_all_models(self):
        """Load all available models"""
        model_names = ["EfficientNetB0", "ResNet50", "DenseNet121"]
        loaded_models = []
        
        for model_name in model_names:
            try:
                self.load_model(model_name)
                loaded_models.append(model_name)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        
        return loaded_models
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms
        transformed = self.transform(image=img)
        img_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(self.device)
    
    def predict(self, image_path, model_name="EfficientNetB0"):
        """Make prediction on a single image"""
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        result = {
            'predicted_class': self.classes[predicted.item()],
            'confidence': confidence.item(),
            'all_probabilities': {
                self.classes[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'model_used': model_name
        }
        
        return result
    
    def ensemble_predict(self, image_path, models_to_use=None):
        """Make ensemble prediction using multiple models"""
        if models_to_use is None:
            models_to_use = ["EfficientNetB0", "ResNet50", "DenseNet121"]
        
        # Load models if not already loaded
        for model_name in models_to_use:
            if model_name not in self.models:
                try:
                    self.load_model(model_name)
                except Exception as e:
                    print(f"Skipping {model_name}: {e}")
                    models_to_use.remove(model_name)
        
        if not models_to_use:
            raise ValueError("No models available for ensemble prediction")
        
        # Preprocess image once
        img_tensor = self.preprocess_image(image_path)
        
        # Collect predictions from all models
        ensemble_probs = torch.zeros(self.num_classes).to(self.device)
        individual_predictions = {}
        
        for model_name in models_to_use:
            model = self.models[model_name]
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                ensemble_probs += probabilities[0]
                
                confidence, predicted = torch.max(probabilities, 1)
                individual_predictions[model_name] = {
                    'predicted_class': self.classes[predicted.item()],
                    'confidence': confidence.item()
                }
        
        # Average the probabilities
        ensemble_probs /= len(models_to_use)
        final_confidence, final_predicted = torch.max(ensemble_probs, 0)
        
        result = {
            'predicted_class': self.classes[final_predicted.item()],
            'confidence': final_confidence.item(),
            'ensemble_probabilities': {
                self.classes[i]: prob.item() 
                for i, prob in enumerate(ensemble_probs)
            },
            'individual_predictions': individual_predictions,
            'models_used': models_to_use
        }
        
        return result