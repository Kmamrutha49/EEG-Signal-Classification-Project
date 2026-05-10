import pandas as pd

def load_eeg_dataset(csv_path):
    data = pd.read_csv(csv_path)

    X = data.drop('label', axis=1).values
    y = data['label'].values

    return X, y
