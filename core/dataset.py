import pandas as pd

class DatasetParser:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None

    def load_dataset(self):
        """Charger le dataset CSV."""
        self.data = pd.read_csv(self.dataset_path)
        return self.data

    def parse_dataset(self):
        """Séparer les variables explicatives et la cible."""
        if self.data is None:
            self.load_dataset()

        X = self.data.drop(columns=["infection"])
        y = self.data["infection"]
        return X, y
