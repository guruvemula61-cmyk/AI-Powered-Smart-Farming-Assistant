class DiseaseDetector:
    """
    Crop disease detection module.

    Designed to connect with an on-device AI vision model
    for real-time crop disease identification.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model_loaded = False

    def detect(self, image):
        """
        Analyze a crop image for possible disease.

        The AI model will be connected in the next stage.
        """

        if image is None:
            return {
                "status": "WAITING",
                "disease": "NO_IMAGE",
                "confidence": 0.0
            }

        if not self.model_loaded:
            return {
                "status": "MODEL_READY",
                "disease": "MODEL_NOT_CONNECTED",
                "confidence": 0.0
            }

        return {
            "status": "ANALYZED",
            "disease": "UNKNOWN",
            "confidence": 0.0
        }
