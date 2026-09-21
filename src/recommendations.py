class FarmingRecommendationEngine:
    """
    Combines crop health, disease, environmental,
    and irrigation results into simple farmer guidance.
    """

    def generate(self, crop_result, disease_result, environment_result, irrigation_result):
        recommendations = []

        # Crop health
        if crop_result.get("status") != "READY":
            recommendations.append(
                "Review the detected crop health condition."
            )

        # Disease
        disease = disease_result.get("disease")

        if disease and disease != "MODEL_NOT_CONNECTED":
            recommendations.append(
                f"Possible crop disease detected: {disease}."
            )

        # Environmental conditions
        recommendations.extend(
            environment_result.get("recommendations", [])
        )

        # Irrigation
        irrigation_action = irrigation_result.get("action")

        if irrigation_action:
            recommendations.append(irrigation_action)

        if not recommendations:
            recommendations.append(
                "No immediate action is required. Continue monitoring the crop."
            )

        return {
            "recommendations": recommendations,
            "priority": irrigation_result.get("priority", "LOW")
        }
