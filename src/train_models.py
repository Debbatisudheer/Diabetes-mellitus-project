from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

def train_models(X_train, y_train):
    # Define models
    rf_grid_params = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    gb_grid_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    }

    # Hyperparameter tuning for Random Forest
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid_params, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    # Hyperparameter tuning for Gradient Boosting
    gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_grid_params, cv=5, scoring='accuracy', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    best_gb = gb_grid.best_estimator_

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": best_rf,
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": best_gb,
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    return models
