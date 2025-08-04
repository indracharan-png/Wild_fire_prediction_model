import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from data_utils import load_data
from plot_utils import plot_roc_curve

# Load preprocessed data
X, y = load_data()

def train_and_test_model():
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100, # no. of trees in the ensemble
        max_depth=6, # maximum depth of a each decision tree
        learning_rate=0.1, # step size shrinkage
        use_label_encoder=False, # avoid warning
        eval_metric='logloss', # evaluation metric for testing set
        random_state=42
    )

    model.fit(X_train, y_train)

    # Get predicted probabilities for ROC-AUC
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)

    # Evaluate with metrics
    print("XGBoost Classifier Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))
    print("\nClassification Report:\n", classification_report(y_test, y_preds))
    auc_score = roc_auc_score(y_test, y_probs)
    print(f"\nROC AUC Score: {auc_score:.4f}")

    # Plot the ROC curve
    plot_roc_curve(y_test, y_probs, title="XGBoost ROC Curve")

    # return the results for models comparison
    results = ("XGBoost model", y_test, y_probs)
    return results