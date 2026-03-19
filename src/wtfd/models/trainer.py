from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class WindFailureModel:
    def __init__(self, model_type='xgboost', params=None):
        if model_type == 'xgboost':
            # scale_pos_weight is key for our 3-day window imbalance
            self.model = XGBClassifier(**(params or {'n_estimators': 100, 'scale_pos_weight': 20}))
        elif model_type == 'rf':
            self.model = RandomForestClassifier(**(params or {'n_estimators': 100, 'class_weight': 'balanced'}))
        else:
            self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # We usually want probabilities for the 48hr failure risk
        return self.model.predict_proba(X_test)[:, 1]