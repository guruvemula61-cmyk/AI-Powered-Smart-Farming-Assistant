"""
Streamlit demo for the AI-Powered Smart Farming Assistant.
 
Run with: streamlit run demo_ui.py
"""
 
import streamlit as st
from PIL import Image
 
from src.disease_detection import DiseaseDetector
from src.environment import EnvironmentalAnalyzer
from src.irrigation import IrrigationAdvisor
from src.recommendations import FarmingRecommendationEngine
from src.crop_health import CropHealthAnalyzer
 
st.set_page_config(page_title="Smart Farming Assistant", page_icon="🌾")
 
st.title("🌾 AI-Powered Smart Farming Assistant")
st.caption("On-device AI crop disease detection - optimized for Snapdragon NPU")
 
detector = DiseaseDetector()
crop_analyzer = CropHealthAnalyzer()
environment_analyzer = EnvironmentalAnalyzer()
irrigation_advisor = IrrigationAdvisor()
recommendation_engine = FarmingRecommendationEngine()
 
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("1. Crop Leaf Image")
    uploaded_file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
 
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded leaf", use_column_width=True)
 
with col2:
    st.subheader("2. Environmental Readings")
    temperature = st.slider("Temperature (°C)", 10, 50, 32)
    humidity = st.slider("Humidity (%)", 0, 100, 55)
    soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 28)
 
st.divider()
 
if st.button("🔍 Analyze", type="primary"):
    disease_result = detector.detect(image)
    crop_result = crop_analyzer.analyze(image)
 
    environment_result = environment_analyzer.analyze(temperature, humidity, soil_moisture)
    irrigation_result = irrigation_advisor.recommend(soil_moisture, temperature)
 
    final_result = recommendation_engine.generate(
        crop_result, disease_result, environment_result, irrigation_result
    )
 
    st.subheader("Results")
 
    if disease_result["status"] == "MODEL_NOT_TRAINED":
        st.warning(disease_result["message"])
    elif disease_result["status"] == "WAITING":
        st.info("Upload a leaf image to run disease detection.")
    else:
        st.metric("Detected Disease", disease_result["disease"],
                   f"{disease_result['confidence'] * 100:.1f}% confidence")
 
    st.write("**Irrigation:**", irrigation_result["action"],
              f"(Priority: {irrigation_result['priority']})")
 
    st.write("**Farmer Recommendations:**")
    for rec in final_result["recommendations"]:
        st.write(f"- {rec}")
 



