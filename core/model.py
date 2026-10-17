
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict_proba(self, patient_data):
        pass
