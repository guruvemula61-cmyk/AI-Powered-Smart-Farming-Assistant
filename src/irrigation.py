class IrrigationAdvisor:
    """
    Irrigation recommendation module.

    Uses soil moisture and environmental conditions
    to generate a basic irrigation recommendation.
    """

    def recommend(self, soil_moisture, temperature):
        if soil_moisture < 25:
            action = "IRRIGATION REQUIRED"
            priority = "HIGH"

        elif soil_moisture < 40:
            action = "IRRIGATION MAY BE NEEDED"
            priority = "MEDIUM"

        else:
            action = "NO IMMEDIATE IRRIGATION REQUIRED"
            priority = "LOW"

        if temperature > 35 and soil_moisture < 40:
            action = "IRRIGATION REQUIRED - HIGH HEAT"
            priority = "HIGH"

        return {
            "action": action,
            "priority": priority,
            "soil_moisture": soil_moisture,
            "temperature": temperature
        }
