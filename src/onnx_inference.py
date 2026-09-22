"""
On-device inference using ONNX Runtime.
 
On a Snapdragon-powered HP PC, this will use the QNN (Qualcomm
Neural Network) execution provider to run inference on the NPU.
On a regular dev machine (no NPU), it automatically falls back to
the CPU execution provider - so this script works everywhere, but
only accelerates on Snapdragon hardware.
"""
 
import os
 
import numpy as np
import onnxruntime as ort
from PIL import Image
 
 
ONNX_MODEL_PATH = os.path.join("models", "mobilenetv3_crop_disease.onnx")
CLASSES_PATH = os.path.join("models", "classes.txt")
IMAGE_SIZE = 224
 
 
def load_classes():
    with open(CLASSES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]
 
 
def get_session(model_path=ONNX_MODEL_PATH):
    """
    Creates an ONNX Runtime session, preferring the Qualcomm NPU
    (QNN) execution provider when available, and falling back to
    CPU otherwise - so the same code runs on any machine.
    """
    available = ort.get_available_providers()
 
    providers = []
    if "QNNExecutionProvider" in available:
        providers.append(("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}))
    providers.append("CPUExecutionProvider")
 
    session = ort.InferenceSession(model_path, providers=providers)
    active_provider = session.get_providers()[0]
    print(f"Running inference on: {active_provider}")
 
    return session
 
 
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))  # HWC -> CHW
    array = np.expand_dims(array, axis=0)   # add batch dim
    return array
 
 
def predict(session, image_path, classes):
    input_name = session.get_inputs()[0].name
    input_tensor = preprocess(image_path)
 
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
 
    exp = np.exp(logits - np.max(logits))
    probabilities = exp / exp.sum()
 
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
 
    return {
        "disease": classes[predicted_idx],
        "confidence": round(confidence, 4)
    }
 
 
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 2:
        print("Usage: python src/onnx_inference.py <path_to_leaf_image>")
        sys.exit(1)
 
    classes = load_classes()
    session = get_session()
    result = predict(session, sys.argv[1], classes)
 
    print(f"Disease    : {result['disease']}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
 



