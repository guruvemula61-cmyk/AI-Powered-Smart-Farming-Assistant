from src.crop_health import CropHealthAnalyzer
from src.disease_detection import DiseaseDetector
from src.environment import EnvironmentalAnalyzer
from src.irrigation import IrrigationAdvisor
from src.recommendations import FarmingRecommendationEngine


def main():
    print("=" * 60)
    print("       AI-POWERED SMART FARMING ASSISTANT")
    print("=" * 60)

    # Initialize modules
    crop_analyzer = CropHealthAnalyzer()
    disease_detector = DiseaseDetector()
    environment_analyzer = EnvironmentalAnalyzer()
    irrigation_advisor = IrrigationAdvisor()
    recommendation_engine = FarmingRecommendationEngine()

    # Sample environmental inputs
    temperature = 32
    humidity = 55
    soil_moisture = 28

    # Crop analysis placeholder
    crop_result = crop_analyzer.analyze(None)

    # Disease analysis placeholder
    disease_result = disease_detector.detect(None)

    # Environmental analysis
    environment_result = environment_analyzer.analyze(
        temperature,
        humidity,
        soil_moisture
    )

    # Irrigation recommendation
    irrigation_result = irrigation_advisor.recommend(
        soil_moisture,
        temperature
    )

    # Final farming recommendations
    final_result = recommendation_engine.generate(
        crop_result,
        disease_result,
        environment_result,
        irrigation_result
    )

    print("\nEnvironmental Conditions")
    print(f"Temperature   : {temperature} °C")
    print(f"Humidity      : {humidity} %")
    print(f"Soil Moisture : {soil_moisture} %")

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
