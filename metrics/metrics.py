import numpy as np
import pandas as pd
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

    # Ensure the tensors are flattened to 1D arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if y_prob is not None:
        y_prob = y_prob.flatten()

    # Use scikit-learn's confusion_matrix for correct counting
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extract values from confusion matrix (sklearn order: [[TN, FP], [FN, TP]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

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
    weighted_error = (fp_cost * fp + fn_cost * fn) / len(y_true) if len(
        y_true) > 0 else 0  # Calculate AUC-ROC if probabilities are provided
    auc_roc = None
    if y_prob is not None:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
        auc_roc = auc(fpr_curve, tpr_curve)

    # Create metrics dictionary
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,  # Spam Catch Rate
        'f1_score': f1_score,
        'specificity': specificity,  # Ham Preservation Rate
        'fpr': fpr,
        'fnr': fnr,
        'auc_roc': auc_roc,
        'youden_j': youden_j,
        'weighted_error': weighted_error
    }

    # Create confusion matrix dictionary
    confusion_dict = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    # Convert metrics to DataFrame for better display in notebooks
    metrics_df = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'Precision',
            'Recall (Spam Catch Rate)',
            'F1-Score',
            'Specificity (Ham Preservation Rate)',
            'False Positive Rate',
            'False Negative Rate',
            'AUC-ROC',
            'Youden\'s J',
            'Weighted Error'
        ],
        'Value': [
            accuracy,
            precision,
            recall,
            f1_score,
            specificity,
            fpr,
            fnr,
            auc_roc if auc_roc is not None else np.nan,
            youden_j,
            weighted_error
        ],
        'Description': [
            'Proportion of correct predictions (TP + TN) / total',
            'Proportion of true positives among predicted positives (TP / (TP + FP))',
            'Proportion of true positives among actual positives (TP / (TP + FN))',
            'Harmonic mean of precision and recall',
            'Proportion of true negatives among actual negatives (TN / (TN + FP))',
            'Proportion of false positives among actual negatives (FP / (FP + TN))',
            'Proportion of false negatives among actual positives (FN / (FN + TP))',
            'Area under the Receiver Operating Characteristic curve',
            'Recall + Specificity - 1',
            f'Weighted cost ({fp_cost}*FP + {fn_cost}*FN) / total'
        ],
        'Optimal': [
            'Higher better',
            'Higher better',
            'Higher better',
            'Higher better',
            'Higher better',
            'Lower better',
            'Lower better',
            'Higher better',
            'Higher better',
            'Lower better'
        ]
    }).set_index('Metric')

    # Add confusion matrix as a separate attribute to the DataFrame
    metrics_df.attrs['confusion_matrix'] = confusion_dict

    # Also preserve the original dictionary format for backward compatibility
    metrics_df.attrs['metrics_dict'] = metrics_dict

    return metrics_df
