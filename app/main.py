
from core.dataset import DatasetParser


def main():
    # Créer le parseur
    parser = DatasetParser(
        "data/clinical_infection_dataset.csv"
    )

    # Charger le dataset
    data = parser.load_dataset()

    print("Aperçu du dataset :")
    print(data.head())

    # Parser le dataset
    X, y = parser.parse_dataset()

    print("\nDimensions de X :", X.shape)
    print("Dimensions de y :", y.shape)

    print("\nColonnes utilisées :")
    print(X.columns.tolist())

    print("\nRépartition de la cible :")
    print(y.value_counts())


if __name__ == "__main__":
    main()
