from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        results[name] = {"accuracy": acc, "auc": auc, "model": model}
    return results

def best_model(results):
    return max(results, key=lambda m: results[m]['accuracy'])
