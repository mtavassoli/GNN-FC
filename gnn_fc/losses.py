import torch

import torch

class FairnessLoss:
    """
    Model-agnostic fairness loss for GNNs.
    Can be applied to any model's predictions, whether fair or unfair.
    
    The loss combines:
        - Equal Opportunity (EO) loss
        - Statistical Parity (SP) loss
    """

    def __init__(self, eo_coeff: float = 1.0, sp_coeff: float = 1.0):
        """
        Args:
            eo_coeff (float): Weight for Equal Opportunity loss.
            sp_coeff (float): Weight for Statistical Parity loss.
        """
        self.eo_coeff = eo_coeff
        self.sp_coeff = sp_coeff
    
    def get_equal_opportunity_loss(self, labels, sensitive_attr, node_mask, preds):
        """
        Compute Equal Opportunity loss.

        Args:
            labels (Tensor): True labels (0/1) of nodes.
            sensitive_attr (Tensor): Sensitive attribute (0/1) of nodes.
            node_mask (Iterable): Indices of nodes to consider (e.g., train_mask).
            preds (Tensor): Model predictions (probabilities).

        Returns:
            Tensor: Equal Opportunity loss (scalar).
        """
        count_group0 = count_group1 = 0
        pred_group0 = pred_group1 = 0.0

        for i in node_mask:
            if labels[i] == 1 and sensitive_attr[i] == 0:
                pred_group0 += preds[i]
                count_group0 += 1
            elif labels[i] == 1 and sensitive_attr[i] == 1:
                pred_group1 += preds[i]
                count_group1 += 1

        if count_group0 == 0 or count_group1 == 0:
            return torch.tensor(0.0, device=preds.device)

        return torch.abs((pred_group0 / count_group0) - (pred_group1 / count_group1))


    def get_equal_opportunity_loss(self, labels, sensitive_attr, node_mask, preds):
        """
        Compute Equal Opportunity loss.

        Args:
            labels (Tensor): True labels (0/1) of nodes.
            sensitive_attr (Tensor): Sensitive attribute (0/1) of nodes.
            node_mask (Iterable): Indices of nodes to consider (e.g., train_mask).
            preds (Tensor): Model predictions (probabilities).

        Returns:
            Tensor: Equal Opportunity loss (scalar).
        """
        count_group0 = count_group1 = 0
        pred_group0 = pred_group1 = 0.0

        for i in node_mask:
            if labels[i] == 1 and sensitive_attr[i] == 0:
                pred_group0 += preds[i]
                count_group0 += 1
            elif labels[i] == 1 and sensitive_attr[i] == 1:
                pred_group1 += preds[i]
                count_group1 += 1

        if count_group0 == 0 or count_group1 == 0:
            return torch.tensor(0.0, device=preds.device)

        return torch.abs(pred_group0 / count_group0 - pred_group1 / count_group1)

    def get_statistical_parity_loss(self, sensitive_attr, node_mask, preds):
        """
        Compute Statistical Parity loss.

        Args:
            sensitive_attr (Tensor): Sensitive attribute (0/1) of nodes.
            node_mask (Iterable): Indices of nodes to consider (e.g., train_mask).
            preds (Tensor): Model predictions (probabilities).

        Returns:
            Tensor: Statistical Parity loss (scalar).
        """
        count_group0 = count_group1 = 0
        pred_group0 = pred_group1 = 0.0

        for i in node_mask:
            if sensitive_attr[i] == 0:
                pred_group0 += preds[i]
                count_group0 += 1
            elif sensitive_attr[i] == 1:
                pred_group1 += preds[i]
                count_group1 += 1

        if count_group0 == 0 or count_group1 == 0:
            return torch.tensor(0.0, device=preds.device)

        return torch.abs((pred_group0 / count_group0) - (pred_group1 / count_group1))
    
    def __call__(self, labels, sensitive_attr, node_mask, preds):
        """
        Compute the combined fairness loss: EO + SP.

        Args:
            labels (Tensor): True labels (0/1).
            sensitive_attr (Tensor): Sensitive attribute (0/1).
            node_mask (Iterable): Indices of nodes to consider (train/val/test mask).
            preds (Tensor): Model predictions (probabilities).

        Returns:
            Tensor: Weighted sum of EO and SP losses.
        """
        eo_loss = self.get_equal_opportunity_loss(labels, sensitive_attr, node_mask, preds)
        sp_loss = self.get_statistical_parity_loss(sensitive_attr, node_mask, preds)
        return self.eo_coeff * eo_loss + self.sp_coeff * sp_loss
