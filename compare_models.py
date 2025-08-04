import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random_forest_model
import xgboost_model
import mlp_model
import optimized_mlp_model

def compare_models():
    """
    Compare the performance of different models on the same dataset.
    This function trains and evaluates Random Forest, XGBoost, and MLP models,
    and plots their ROC curves for comparison.
    """
    # Train and test each model
    rf_results = random_forest_model.train_and_test_model()
    xgb_results = xgboost_model.train_and_test_model()
    mlp_results = mlp_model.train_and_test_model()
    optimized_mlp_results = optimized_mlp_model.train_and_test_model()

    # Combine results for plotting with traditional mlp model
    traditional_mlp_model_comparison_outputs = [rf_results, xgb_results, mlp_results]

    # Combine results for plotting with optimized mlp model
    optimized_mlp_model_comparison_ouputs = [rf_results, xgb_results, optimized_mlp_results]


    # Plot combined ROC curve
    plot_combined_roc(traditional_mlp_model_comparison_outputs, title="Traditional MLP Model Comparison")
    plot_combined_roc(optimized_mlp_model_comparison_ouputs, title="Optimized MLP Model Comparison")



def plot_combined_roc(models_outputs, title="Combined ROC Curve"):
    """
    Plot ROC curves for multiple models on the same figure.

    Parameters:
    - models_outputs: list of tuples (model_name, y_true, y_probs)
    - title: title of the plot
    """
    plt.figure(figsize=(10, 8))

    for model_name, y_true, y_probs in models_outputs:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()






if __name__ == "__main__":
    compare_models()
