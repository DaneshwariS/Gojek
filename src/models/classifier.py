from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score

class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass

class SklearnClassifier(Classifier):
    def __init__(self, estimator: BaseEstimator, features: List[str], target: str):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        # Get true labels
        y_true = df_test[self.target]
        # Get predicted labels
        y_pred = self.clf.predict(df_test[self.features].values)

        # Compute accuracy and F1 score
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')  # 'weighted' accounts for class imbalance

        # Return the metrics in the required format
        return {
            "accuracy": accuracy,
            "f1_score": f1
        }

    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]
