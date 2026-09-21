class EnvironmentalAnalyzer:
    """
    Environmental condition analysis module.

    This module will process environmental data such as
    temperature, humidity, and soil moisture.
    """

    def analyze(self, temperature, humidity, soil_moisture):
        """
        Analyze basic environmental conditions.
        """

        recommendations = []

        if soil_moisture < 30:
            recommendations.append("Irrigation may be required.")

        if temperature > 35:
            recommendations.append("High temperature detected.")

        if humidity < 30:
            recommendations.append("Low humidity detected.")

        return {
            "temperature": temperature,
            "humidity": humidity,
            "soil_moisture": soil_moisture,
            "recommendations": recommendations
        }
