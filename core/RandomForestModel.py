# RandomForestModel.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from core.model import Model


class RandomForestModel(Model):
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            ))
        ])

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, patient_data):
        return self.model.predict_proba(patient_data)