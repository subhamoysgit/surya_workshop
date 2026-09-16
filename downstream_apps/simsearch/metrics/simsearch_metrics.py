"""
Template metrics for flare forecasting.

FlareMetrics defines four metric sets:
- "train_loss"    — differentiable loss that drives backpropagation (MSE).
- "val_loss"      — the quantity logged as `val_loss` and used to select checkpoints.
                    Defaults to the same MSE as "train_loss"; override it when your task
                    needs a different validation objective.
- "train_metrics" — non-differentiable metrics logged during training (RRSE).
- "val_metrics"   — metrics logged at validation for reporting only (MSE + RRSE). These
                    do NOT influence checkpoint selection — "val_loss" does.

The __call__ method selects the appropriate metric set based on the mode passed at
construction time. The dictionary keys returned by each method become the metric names
propagated to the logger (e.g. WandB, CSV).
"""

import torch
import torchmetrics as tm  # Lots of possible metrics in here https://lightning.ai/docs/torchmetrics/stable/all-metrics.html
import torch.nn.functional as F
# Shape contract: predictions arrive as (B,) from HelioSpectformer1D or (B, 1) from the
# linear baseline, while targets are always (B, 1). Every metric below flattens both with
# reshape(-1) rather than squeeze(-1): squeeze is shape-dependent and collapses a
# batch of one to a 0-d scalar, which then fails to broadcast against a (1,) target.
import torch

def euclidean_distance(preds, target):
    # [batch_size, features] → mean distance across examples
    return torch.linalg.vector_norm(preds - target, dim=-1).mean()


class SimsearchMetrics:
    def __init__(self, mode: str):
        """
        Initialize FlareMetrics class.

        Args:
            mode (str): Mode to use for metric evaluation. One of "train_loss",
                        "val_loss", "train_metrics", or "val_metrics".
        """
        self.mode = mode

        # Cache torchmetrics instances once (instead of recreating each call)
        self._euclid = euclidean_distance

    def _ensure_device(self, preds: torch.Tensor) -> None:
        """Move torchmetrics modules to the same device as ``preds``, if needed."""
        if self._euclid.device != preds.device:
            self._euclid = self._euclid.to(preds.device)

    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated loss metrics.
                                        Keys are metric names (e.g., "mse"), and values are the
                                        corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """

        output_metrics = {}
        output_weights = []

        output_metrics["cosine"] = -F.cosine_similarity(preds, target, dim=-1).mean()
        output_weights.append(1)
        return output_metrics, output_weights

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Evaluate metrics for the mode set at construction time.

        Args:
            preds: Model output tensor. Shape depends on the application.
            target: Ground truth tensor to compare against.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - Metric dictionary. Keys become logger metric names; values are
                  scalar tensors aggregated over the batch.
                - List of per-metric weights (used by FlareLightningModule to
                  combine multiple loss terms into a single scalar).
        """

        match self.mode.lower():

            case "train_loss":
                return self.train_loss(preds, target)

            case _:
                raise NotImplementedError(
                    f"{self.mode} is not implemented as a valid metric case."
                )
