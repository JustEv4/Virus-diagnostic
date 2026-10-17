import pandas as pd


class DatasetParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.target_column = "infection"

    def load(self):
        # Charger le dataset CSV
        data = pd.read_csv(self.file_path)

        # Vérifier la présence de la colonne cible
        if self.target_column not in data.columns:
            raise ValueError(
                f"La colonne '{self.target_column}' est absente du dataset."
            )

        return data

    def parse(self):
        # Charger les données
        data = self.load()

        # Séparer les variables explicatives et la cible
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        return X, y
