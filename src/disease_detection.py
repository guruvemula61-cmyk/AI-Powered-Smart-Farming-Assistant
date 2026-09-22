import os
 
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
 
 
DEFAULT_MODEL_PATH = os.path.join("models", "mobilenetv3_crop_disease.pth")
DEFAULT_CLASSES_PATH = os.path.join("models", "classes.txt")
 
IMAGE_SIZE = 224
 
 
class DiseaseDetector:
    """
    Crop disease detection module.
 
    Loads the trained MobileNetV3-Small model and runs real
    on-device inference on a crop image (file path, PIL Image,
    or numpy array from OpenCV).
    """
 
    def __init__(self, model_path=DEFAULT_MODEL_PATH, classes_path=DEFAULT_CLASSES_PATH):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model_loaded = False
        self.model = None
        self.classes = []
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
 
        self._load_classes()
        self._load_model()
 
    def _load_classes(self):
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r") as f:
                self.classes = [line.strip() for line in f if line.strip()]
 
    def _load_model(self):
        if not os.path.exists(self.model_path) or not self.classes:
            # Model not trained/exported yet - detect() will report this clearly
            self.model_loaded = False
            return
 
        num_classes = len(self.classes)
 
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
 
        self.model = model
        self.model_loaded = True
 
    def _to_pil(self, image):
        """Accepts a file path, PIL Image, or numpy (OpenCV BGR) array."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        # numpy array from OpenCV (BGR) -> RGB PIL image
        import numpy as np
        if isinstance(image, np.ndarray):
            rgb = image[:, :, ::-1]
            return Image.fromarray(rgb)
        raise TypeError("Unsupported image type for detection.")
 
    def detect(self, image):
        """
        Run disease detection on a crop image.
 
        Returns disease name, confidence, and status.
        """
        if image is None:
            return {
                "status": "WAITING",
                "disease": "NO_IMAGE",
                "confidence": 0.0
            }
 
        if not self.model_loaded:
            return {
                "status": "MODEL_NOT_TRAINED",
                "disease": "MODEL_NOT_CONNECTED",
                "confidence": 0.0,
                "message": (
                    "No trained model found at "
                    f"'{self.model_path}'. Run training/train.py first."
                )
            }
 
        pil_image = self._to_pil(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
 
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)
 
        disease_name = self.classes[predicted_idx.item()]
 
        return {
            "status": "ANALYZED",
            "disease": disease_name,
            "confidence": round(confidence.item(), 4)
        }
 
