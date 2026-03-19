from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

class WindFaultTrainer:
    def __init__(self, model_type='xgboost', params=None, random_state=42):
        self.model_type = model_type
        self.params = params or {}
        self.random_state = random_state
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == 'xgboost':
            # XGBoost default: requires manual scale_pos_weight or calculated later
            config = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'binary:logistic',
                'random_state': self.random_state,
                'tree_method': 'hist'
            }
            config.update(self.params)
            return XGBClassifier(**config)
        
        elif self.model_type == 'rf':
            # Random Forest: use 'balanced' to handle the rare failure windows
            config = {
                'n_estimators': 100,
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'n_jobs': -1
            }
            config.update(self.params)
            return RandomForestClassifier(**config)
            
        elif self.model_type == 'logistic':
            # Logistic Regression: simple, fast, and highly interpretable
            config = {
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': self.random_state
            }
            config.update(self.params)
            return LogisticRegression(**config)

    def train(self, X_train, y_train):
        if self.model_type == 'xgboost':
            # Calculate the actual imbalance ratio from the CURRENT training slice
            pos = np.sum(y_train)
            neg = len(y_train) - pos
            
            # dynamic weight: 5,129,822 / 113,126 ≈ 45.3
            calc_weight = neg / pos if pos > 0 else 1
            
            self.model.set_params(
                scale_pos_weight=calc_weight,
                eval_metric='aucpr', # Use Precision-Recall AUC instead of standard AUC
                early_stopping_rounds=None
            )
            
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, preds, output_dict=True)
        
        return {
            'model': self.model_type,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'accuracy': report['accuracy'],
            'auc': roc_auc_score(y_test, probs)
        }