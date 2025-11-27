
import numpy as np

def evaluate_group_fairness(output, idx, labels, sensitive_attr):
    """
    Evaluate group fairness metrics: Statistical Parity and Equality of Opportunity.

    Args:
        output (torch.Tensor): Model predictions (logits or probabilities).
        idx (torch.Tensor): Indices of samples to evaluate.
        labels (torch.Tensor): True binary labels.
        sensitive_attr (torch.Tensor): Sensitive attribute (binary, 0 or 1).

    Returns:
        tuple: (statistical_parity, equality_of_opportunity)
    """
    # Convert tensors to numpy arrays
    true_labels = labels[idx].cpu().numpy()
    sens_vals = sensitive_attr[idx].cpu().numpy()

    # Masks for sensitive groups
    group0_mask = sens_vals == 0
    group1_mask = sens_vals == 1

    # Masks for positive labels within each group (for equality of opportunity)
    group0_pos_mask = np.logical_and(group0_mask, true_labels == 1)
    group1_pos_mask = np.logical_and(group1_mask, true_labels == 1)

    # Binary predictions
    predictions = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()

    # Safe mean to avoid division by zero
    def safe_mean(arr, mask):
        return arr[mask].mean() if mask.sum() > 0 else 0.0

    # Compute fairness metrics
    statistical_parity = abs(safe_mean(predictions, group0_mask) - safe_mean(predictions, group1_mask))
    equality_of_opportunity = abs(safe_mean(predictions, group0_pos_mask) - safe_mean(predictions, group1_pos_mask))

    return statistical_parity, equality_of_opportunity
