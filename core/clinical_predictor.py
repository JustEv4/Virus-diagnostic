
class ClinicalPredictor:

    def __init__(self):
        self.model = None

    def set_model(self, model):
        """Recevoir un modèle déjà entraîné."""
        self.model = model

    def diagnose(self, patient_data):
        """Prédire si le patient est sain ou infecté."""
        if self.model is None:
            raise ValueError("Aucun modèle n'a été fourni.")

        probability = self.model.predict_proba(patient_data)[0][1]

        if probability >= 0.5:
            return "infecté"

        return "sain"
