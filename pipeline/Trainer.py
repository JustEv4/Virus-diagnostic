from sklearn.model_selection import train_test_split
from core.dataset import DatasetParser
from core.LogisticRegressionModel import LogisticRegressionModel

class Trainer:
    def __init__(self, dataset_path):
        # Chemin du dataset
        self.dataset_path = dataset_path

        # Créer le parser du dataset
        self.parser = DatasetParser(dataset_path)

        # Créer le modèle de régression logistique
        self.model = LogisticRegressionModel()

    def train(self, test_size=0.2, random_state=42):
        # 1. Charger et parser le dataset
        X, y = self.parser.parse()

        # 2. Diviser les données en entraînement et test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # 3. Entraîner le modèle sur les données d'entraînement
        self.model.train(X_train, y_train)

        # 4. Retourner le modèle et les données de test
        return self.model, X_test, y_test