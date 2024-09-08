import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef
from boruta import BorutaPy

class fireClfStack():
    def fit(self, X_train, y_train):
        # Определение базовых моделей
        catboost_clf = CatBoostClassifier(verbose=0, random_state=42)
        lgbm_clf = LGBMClassifier(random_state=42)
        svc_clf = LinearSVC(random_state=42)
        dt_clf = DecisionTreeClassifier(random_state=42)
        xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        # Использование Boruta для отбора значимых признаков
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
        boruta_selector.fit(X_train, y_train)

        # Получение значимых признаков
        X_train = boruta_selector.transform(X_train)
        X_test = boruta_selector.transform(X_test)

        # Voting Classifier
        self.voting_clf = VotingClassifier(
            estimators=[
                ('catboost', catboost_clf),
                ('lgbm', lgbm_clf),
                ('svc', svc_clf),
                ('dt', dt_clf),
                ('xgb', xgb_clf)
            ],
            voting='hard'  # можно поменять на 'soft' для усреднения вероятностей
        )

        # Обучение Voting Classifier
        self.voting_clf.fit(X_train, y_train)

    def printMCC(self, X_test, y_test):
        # Предсказание и оценка качества модели
        self.pred = self.voting_clf.predict(X_test)
        print(f'Voting Classifier MCC: {matthews_corrcoef(y_test, self.pred):.4f}')
