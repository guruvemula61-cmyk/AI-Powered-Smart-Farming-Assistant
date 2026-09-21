class DiseaseDetector:
    """
    Crop disease detection module.

    This module will later connect to an AI vision model
    for real-time crop disease identification.
    """

    def detect(self, image):
        """
        Analyze a crop image for possible disease.

        Returns a basic response until the trained
        AI model is integrated.
        """

        return {
            "status": "READY",
            "disease": "MODEL_NOT_CONNECTED",
            "confidence": 0.0
        }
