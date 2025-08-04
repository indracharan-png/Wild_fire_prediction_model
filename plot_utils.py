import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np


def plot_roc_curve(y_true, y_probs, title="ROC Curve", save_path=None):
    """
    Plots the ROC curve for a binary classifier.
    
    Parameters:
    - y_true: list or numpy array of true binary labels (0 or 1)
    - y_probs: list or array of predicted probabilities (floats between 0 and 1)
    - title: optional plot title
    - save_path: if provided, saves the figure to this path
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()



def plot_precision_recall_curve(y_true, y_scores, title="Precision-Recall Curve", save_path=None):
    """
    Plots the Precision-Recall curve for a binary classifier.
    
    Parameters:
    - y_true: list or numpy array of true binary labels (0 or 1)
    - y_scores: list or array of predicted probabilities (floats between 0 and 1)
    - title: optional plot title
    - save_path: if provided, saves the figure to this path
    """
    

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-Recall curve saved to {save_path}")
    else:
        plt.show()  



def plot_threshold_metrics(y_true, y_preds, title="MLP Model ROC Curve"):
    """
    Plots Precision, Recall, and F1 Score against different thresholds.
    
    Parameters:
    - y_true: list or numpy array of true binary labels (0 or 1)
    - y_preds: list or array of predicted probabilities (floats between 0 and 1)
    """
    
    # Compute precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_preds)

    # Avoid threshold mismatch (precision and recall have len = len(thresholds)+1)
    thresholds = np.append(thresholds, 1.0)

    # Compute F1 scores
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Plot all three metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, label='Precision', color='b')
    plt.plot(thresholds, recall, label='Recall', color='g')
    plt.plot(thresholds, f1, label='F1 Score', color='r')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_loss_curve(losses, title="Loss vs Epochs", save_path=None):
    """
    Plots the training loss over epochs.

    Parameters:
    - losses: list or array of loss values (one per epoch)
    - title: optional plot title
    - save_path: if provided, saves the figure to this path
    """

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
    else:
        plt.show()
