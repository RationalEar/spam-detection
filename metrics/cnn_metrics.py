import numpy as np
import torch
from sklearn.metrics import roc_curve, auc


def compute_metrics(y_true, y_pred, y_prob=None, fp_cost=0.3, fn_cost=0.7):
    """
    Compute comprehensive evaluation metrics including AUC-ROC, FPR, FNR,
    Accuracy, Precision, Recall, F1-Score, Specificity (Ham Preservation Rate),
    Youden's J, and cost-sensitive evaluation.
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities (for AUC-ROC)
        fp_cost: Cost weight for false positives
        fn_cost: Cost weight for false negatives
    Returns:
        dict: Dictionary containing all metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    # Calculate confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate basic metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    # Additional performance metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as Sensitivity or Spam Catch Rate
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Same as Ham Preservation Rate (1 - FPR)
    youden_j = recall + specificity - 1

    # Calculate cost-sensitive error
    weighted_error = (fp_cost * fp + fn_cost * fn) / len(y_true) if len(y_true) > 0 else 0

    # Calculate AUC-ROC if probabilities are provided
    auc_roc = None
    if y_prob is not None:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
        auc_roc = auc(fpr_curve, tpr_curve)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,  # Spam Catch Rate
        'f1_score': f1_score,
        'specificity': specificity,  # Ham Preservation Rate
        'fpr': fpr,
        'fnr': fnr,
        'auc_roc': auc_roc,
        'youden_j': youden_j,
        'weighted_error': weighted_error,
        'confusion_matrix': {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    }
