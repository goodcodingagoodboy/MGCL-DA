import torch
import torch.nn as nn


def custom_loss_function(avg_output, group_outputs, target, param, alpha=0.5, beta=0.3, return_contrast_loss=False):
    """
    Custom loss function combining classification cross-entropy loss and feature cosine similarity.

    Args:
        avg_output: Model output logits for classification.
        group_outputs: Three sets of network feature outputs.
        target: Ground truth labels (used for classification cross-entropy loss).
        param: Overall weight coefficient for cosine similarity loss.
        alpha: Weight controlling similarity between original and self-aware features (default: 0.5).
        beta: Weight controlling similarity between cross-sample and self-aware features (default: 0.3).
        return_contrast_loss: Whether to return the contrastive loss separately (default: False).

    Returns:
        If return_contrast_loss=False:
            Returns total_loss: Combination of classification loss and weighted cosine similarity loss.
        If return_contrast_loss=True:
            Returns (total_loss, contrast_loss): Tuple containing total loss and contrastive loss.
    """
    # Ensure target labels are within the range [0, 1]
    target = torch.clamp(target.view(-1, 1).float(), 0, 1)

    # Use BCEWithLogitsLoss, ensuring the input is logits rather than sigmoid outputs
    classification_loss = nn.BCEWithLogitsLoss()(avg_output, target)

    # Extract the three feature groups
    cross_sample = group_outputs[0]
    original = group_outputs[1]
    self_aware = group_outputs[2]

    # Compute cosine similarity between different feature pairs
    cosine_similarity_12 = nn.functional.cosine_similarity(cross_sample, original, dim=1).mean()
    cosine_similarity_13 = nn.functional.cosine_similarity(cross_sample, self_aware, dim=1).mean()
    cosine_similarity_23 = -(nn.functional.cosine_similarity(original, self_aware, dim=1).mean())

    # Compute the weighted cosine similarity loss using alpha and beta
    contrast_loss = (1 - alpha) * cosine_similarity_12 + beta * cosine_similarity_13 + alpha * cosine_similarity_23

    # Compute the total loss
    total_loss = classification_loss + param * contrast_loss

    if return_contrast_loss:
        return total_loss, contrast_loss
    else:
        return total_loss