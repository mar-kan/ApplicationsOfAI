from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch


class TabNet:
    def __init__(self, n_d=8, n_steps=3, gamma=1.3, lambda_sparse=1e-4, patience=10, max_epochs=100):
        self.model = TabNetClassifier(
            n_d=n_d,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            verbose=1
        )

        self.patience = patience
        self.max_epochs = max_epochs

    def fit(self, X_train, y_train, X_val, y_val, batch_size=128, class_weights=None):
        """ Fits the model using native pytorch-tabnet API """

        self.model.fit(
            X_train=X_train.to_numpy(),
            y_train=y_train.to_numpy(),
            eval_set=[(X_val.to_numpy(), y_val.to_numpy())],
            eval_name=['valid'],
            eval_metric=['balanced_accuracy'],
            max_epochs=self.max_epochs,
            patience=self.patience,
            weights=class_weights,
            batch_size=batch_size,
        )

    def predict_and_evaluate(self, X_test, y_test):
        """ Returns predictions, probabilities, and a dictionary of metrics """

        y_pred = self.model.predict(X_test.to_numpy())
        y_prob = self.model.predict_proba(X_test.to_numpy())
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro'),
            "f1": f1_score(y_test, y_pred, average='macro'),
            "roc_auc": roc_auc_score(y_test, y_prob, multi_class='ovr')
        }
        return y_pred, y_prob, metrics
