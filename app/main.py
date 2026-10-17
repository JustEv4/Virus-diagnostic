
import pandas as pd

from pipeline.Trainer import Trainer
from pipeline.ClinicalPredictor import ClinicalPredictor


def main():
    # 1. Créer et utiliser le Trainer
    trainer = Trainer("data/clinical_infection_dataset.csv")

    model, X_test, y_test = trainer.train()

    print("Modèle entraîné avec succès !")
    print("Nombre de patients de test :", len(X_test))

    # 2. Définir le seuil à l'extérieur du Predictor
    threshold = 0.5

    # 3. Créer le prédicteur
    predictor = ClinicalPredictor(model, threshold)

    # 4. Créer un patient fictif
    patient = pd.DataFrame([{
        "age": 45,
        "temperature": 38.6,
        "heart_rate": 110,
        "respiratory_rate": 23,
        "spo2": 96,
        "wbc": 14.5,
        "crp": 85.0,
        "symptoms": 1
    }])

    # Respecter les colonnes utilisées pendant l'entraînement
    patient = patient[X_test.columns]

    # 5. Effectuer la prédiction
    probability = model.predict_proba(patient)[0][1]
    diagnosis = predictor.diagnose(patient)

    print("Probabilité d'infection :", round(probability, 3))
    print("Diagnostic prédit :", diagnosis)


if __name__ == "__main__":
    main()
