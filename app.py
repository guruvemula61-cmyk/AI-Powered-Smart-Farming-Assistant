import argparse
 
import cv2
 
from src.crop_health import CropHealthAnalyzer
from src.disease_detection import DiseaseDetector
from src.environment import EnvironmentalAnalyzer
from src.irrigation import IrrigationAdvisor
from src.recommendations import FarmingRecommendationEngine
 
 
def capture_from_webcam():
    """Opens the default webcam and captures a single frame on SPACE key."""
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture a leaf image, ESC to cancel.")
 
    frame = None
    while True:
        ok, live_frame = cap.read()
        if not ok:
            break
 
        cv2.imshow("Smart Farming Assistant - Press SPACE to capture", live_frame)
        key = cv2.waitKey(1)
 
        if key % 256 == 32:  # SPACE
            frame = live_frame
            break
        if key % 256 == 27:  # ESC
            break
 
    cap.release()
    cv2.destroyAllWindows()
    return frame
 
 
def main():
    parser = argparse.ArgumentParser(description="AI-Powered Smart Farming Assistant")
    parser.add_argument("--image", type=str, default=None, help="Path to a crop leaf image")
    parser.add_argument("--webcam", action="store_true", help="Capture image from webcam instead")
    parser.add_argument("--temperature", type=float, default=32)
    parser.add_argument("--humidity", type=float, default=55)
    parser.add_argument("--soil-moisture", type=float, default=28)
    args = parser.parse_args()
 
    print("=" * 60)
    print("       AI-POWERED SMART FARMING ASSISTANT")
    print("=" * 60)
 
    crop_analyzer = CropHealthAnalyzer()
    disease_detector = DiseaseDetector()
    environment_analyzer = EnvironmentalAnalyzer()
    irrigation_advisor = IrrigationAdvisor()
    recommendation_engine = FarmingRecommendationEngine()
 
    # Resolve the crop image: webcam capture, file path, or none
    image = None
    if args.webcam:
        image = capture_from_webcam()
    elif args.image:
        image = args.image
 
    crop_result = crop_analyzer.analyze(image)
    disease_result = disease_detector.detect(image)
 
    environment_result = environment_analyzer.analyze(
        args.temperature,
        args.humidity,
        args.soil_moisture
    )
 
    irrigation_result = irrigation_advisor.recommend(
        args.soil_moisture,
        args.temperature
    )
 
    final_result = recommendation_engine.generate(
        crop_result,
        disease_result,
        environment_result,
        irrigation_result
    )
 
    print("\nEnvironmental Conditions")
    print(f"Temperature   : {args.temperature} °C")
    print(f"Humidity      : {args.humidity} %")
    print(f"Soil Moisture : {args.soil_moisture} %")
 
    print("\nDisease Detection")
    print(f"Status     : {disease_result['status']}")
    print(f"Disease    : {disease_result['disease']}")
    print(f"Confidence : {disease_result['confidence'] * 100:.2f}%")
 
    print("\nIrrigation Recommendation")
    print(f"Action   : {irrigation_result['action']}")
    print(f"Priority : {irrigation_result['priority']}")
 
    print("\nFarmer Recommendations")
    for recommendation in final_result["recommendations"]:
        print(f"- {recommendation}")
 
    print("\nSystem Status: READY")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 
