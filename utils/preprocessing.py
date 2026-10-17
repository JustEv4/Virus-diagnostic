
def preprocess_data(X):
    # Remplacer les valeurs manquantes par la médiane
    X = X.copy()
    X = X.fillna(X.median())

    return X
