from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from core.model import Model

class LogisticRegressionModel(Model):
    def __init__(self):
        # Prétraitement + modèle de classification
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ])

    def train(self, X, y):
        # Entraîner le modèle
        self.model.fit(X, y)
        return self

    def predict_proba(self, patient_data):
        # Retourner les probabilités des deux classes
        return self.model.predict_proba(patient_data)
