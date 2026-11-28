class ClinicalPredictor:
    def __init__(self, model, threshold):
        # Receive an already trained model and an external threshold
        self.model = model
        self.threshold = threshold

    def diagnose(self, patient_data):
        # Get the probability of class 1 (infected)
        probability = self.model.predict_proba(patient_data)[0][1]

        # Apply the threshold
        if probability >= self.threshold:
            return "infecté"

        return "sain"
