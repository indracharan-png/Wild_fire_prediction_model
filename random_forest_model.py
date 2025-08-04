from data_utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from plot_utils import plot_roc_curve

X, y = load_data()

def train_and_test_model():

  # Train-test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  # Create and train the Random Forest model
  model = RandomForestClassifier(n_estimators=100, random_state=42)

  # Fit the model
  model.fit(X_train, y_train)

  # Predict and evaluate 
  y_preds = model.predict(X_test)
  y_probs = model.predict_proba(X_test)[:, 1] # For ROC AUC score

  print("Random Forest Classifier Results:")
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))
  print("\nClassification Report:\n", classification_report(y_test, y_preds))
  print(f"\nROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")

  plot_roc_curve(y_test, y_probs, title="Random Forest ROC Curve")

  # return the results for models comparison
  results = ("Random Forest model", y_test, y_probs)
  return results
