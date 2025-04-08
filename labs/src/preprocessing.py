import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import yaml
import joblib
import os

params = yaml.safe_load(open("params.yaml"))["features"]

def load_data():
    data = pd.read_csv("data/heart_disease_uci.csv")
    return data

def preprocess_categorical(data):
    data = data.copy()
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    for col in ['fbs', 'exang']:
        data[col] = data[col].fillna('FALSE')
        data[col] = data[col].map({'TRUE': 1, 'FALSE': 0})

    le = LabelEncoder()
    for col in categorical_columns:
        if col not in ['fbs', 'exang']:
            if data[col].isnull().any():
                most_common = data[col].mode()[0]
                data[col] = data[col].fillna(most_common)
            data[col] = le.fit_transform(data[col])

    return data

def create_preprocessing_pipeline():
    return Pipeline([
        ('scaler', StandardScaler())
    ])

def main():
    os.makedirs("data/processed", exist_ok=True)
    data = load_data()
    data = preprocess_categorical(data)
    X = data.drop(['id', 'dataset', 'num'], axis=1)
    y = data['num']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = create_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    train_processed = pd.DataFrame(
        X_train_processed,
        columns=X_train.columns,
        index=X_train.index
    )
    test_processed = pd.DataFrame(
        X_test_processed,
        columns=X_test.columns,
        index=X_test.index
    )
    train_processed['target'] = y_train
    test_processed['target'] = y_test
    train_processed.to_csv("data/processed/train.csv", index=False)
    test_processed.to_csv("data/processed/test.csv", index=False)
    joblib.dump(pipeline, "data/processed/preprocessing_pipeline.joblib")

if __name__ == "__main__":
    main()
