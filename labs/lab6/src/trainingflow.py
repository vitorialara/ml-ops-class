from metaflow import FlowSpec, step, Parameter, card
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from metaflow import current

class TrainingFlow(FlowSpec):
    """
    Training flow for machine learning model development and evaluation.
    This flow implements a complete ML pipeline including data preprocessing,
    model training, and performance evaluation using MLFlow for experiment tracking.
    """

    # Model configuration parameters
    test_size = Parameter('test_size',
                         help='Proportion of dataset to include in test split',
                         default=0.2)

    random_state = Parameter('random_state',
                           help='Random seed for reproducibility',
                           default=42)

    n_estimators = Parameter('n_estimators',
                           help='Number of trees in the random forest',
                           default=100)

    cv_folds = Parameter('cv_folds',
                        help='Number of cross-validation folds',
                        default=5)

    @step
    def start(self):
        """
        Data ingestion and preprocessing step.
        Loads the dataset, performs train-test split, and applies feature scaling.
        """
        # Synthetic dataset generation for demonstration
        # In a real scenario, this would be replaced with actual data loading
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=2,
            n_informative=15,
            n_redundant=5,
            random_state=self.random_state
        )

        # Train-test split with stratification to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Feature scaling for better model performance
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # Store data shapes for reference
        self.train_shape = self.X_train.shape
        self.test_shape = self.X_test.shape

        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Model training and evaluation step.
        Trains a Random Forest classifier and logs performance metrics to MLFlow.
        """
        # MLFlow connection with retry mechanism
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                mlflow.set_tracking_uri("http://localhost:5001")
                mlflow.set_experiment("metaflow-lab6-vitoria")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"MLFlow connection failed after {max_retries} attempts: {str(e)}")
                print(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        with mlflow.start_run():
            # Model initialization and training
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1  # Utilize all available cores
            )

            # Cross-validation
            cv_scores = cross_val_score(
                self.model,
                self.X_train,
                self.y_train,
                cv=self.cv_folds,
                scoring='accuracy'
            )
            self.cv_mean = cv_scores.mean()
            self.cv_std = cv_scores.std()

            # Final model training
            self.model.fit(self.X_train, self.y_train)

            # Model evaluation
            y_pred = self.model.predict(self.X_test)
            self.accuracy = accuracy_score(self.y_test, y_pred)
            self.precision = precision_score(self.y_test, y_pred)
            self.recall = recall_score(self.y_test, y_pred)
            self.f1 = f1_score(self.y_test, y_pred)

            # Feature importance visualization
            feature_importance = pd.DataFrame({
                'feature': range(self.X_train.shape[1]),
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Feature Importance')
            plt.tight_layout()
            self.feature_importance_plot = plt.gcf()

            # Log model parameters and performance metrics
            mlflow.log_params({
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "test_size": self.test_size,
                "cv_folds": self.cv_folds
            })

            mlflow.log_metrics({
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1,
                "cv_mean": self.cv_mean,
                "cv_std": self.cv_std
            })

            # Log the model
            mlflow.sklearn.log_model(
                self.model,
                "random_forest_model",
                registered_model_name="metaflow-rf-model"
            )

        self.next(self.end)

    @card
    @step
    def end(self):
        """
        Final step to display model performance metrics and visualizations.
        """
        from metaflow.cards import Markdown, Image

        # Initialize card
        self.card = current.card

        # Create performance metrics card
        metrics_md = f"""
        # Model Training Results

        ## Dataset Information
        - Training set shape: {self.train_shape}
        - Test set shape: {self.test_shape}

        ## Model Performance
        - Accuracy: {self.accuracy:.4f}
        - Precision: {self.precision:.4f}
        - Recall: {self.recall:.4f}
        - F1 Score: {self.f1:.4f}

        ## Cross-Validation Results
        - Mean CV Score: {self.cv_mean:.4f}
        - CV Standard Deviation: {self.cv_std:.4f}
        """

        # Save feature importance plot
        self.feature_importance_plot.savefig('feature_importance.png')

        # Add content to card
        self.card.append(Markdown(metrics_md))
        self.card.append(Image('feature_importance.png'))

        print("\nModel Training Results:")
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"F1 Score: {self.f1:.4f}")
        print(f"CV Mean Score: {self.cv_mean:.4f}")
        print(f"CV Standard Deviation: {self.cv_std:.4f}")
        print("\nModel successfully registered in MLFlow")

if __name__ == '__main__':
    TrainingFlow()