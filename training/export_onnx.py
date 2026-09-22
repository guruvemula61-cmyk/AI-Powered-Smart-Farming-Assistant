"""
Exports the trained MobileNetV3-Small crop disease model to ONNX
so it can be deployed on Snapdragon NPUs via the Qualcomm AI Hub /
QNN (Qualcomm Neural Network) execution provider.
 
Workflow:
1. Train on a regular x86 machine (training/train.py)
2. Export to ONNX (this script)
3. On the Snapdragon HP device, run inference with ONNX Runtime
   using the QNN execution provider (src/onnx_inference.py)
"""
 
import os
 
import torch
from torch import nn
from torchvision import models
 
 
MODEL_PATH = os.path.join("models", "mobilenetv3_crop_disease.pth")
ONNX_OUTPUT = os.path.join("models", "mobilenetv3_crop_disease.onnx")
CLASSES_PATH = os.path.join("models", "classes.txt")
 
IMAGE_SIZE = 224
 
 
def load_classes():
    with open(CLASSES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]
 
 
def main():
    classes = load_classes()
    num_classes = len(classes)
 
    device = torch.device("cpu")  # export on CPU is fine
 
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
 
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
 
    os.makedirs("models", exist_ok=True)
 
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
 
    print(f"ONNX model exported to: {ONNX_OUTPUT}")
    print(
        "Next: use Qualcomm AI Hub (app.aihub.qualcomm.com) to compile/optimize "
        "this ONNX model for the target Snapdragon NPU, then run it on-device "
        "with ONNX Runtime + QNN Execution Provider (see src/onnx_inference.py)."
    )
 
 
if __name__ == "__main__":
    main()
 



