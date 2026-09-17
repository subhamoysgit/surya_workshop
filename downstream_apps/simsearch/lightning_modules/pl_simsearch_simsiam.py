"""
pl_simple_baseline.py

A minimal PyTorch Lightning wrapper for training a flare prediction model.

This module defines a single LightningModule (FlareLightningModule) that:
  - Calls a user-provided PyTorch model on batched inputs (batch["ts"])
  - Computes one or more training/validation losses via a user-provided loss function
  - Logs scalar losses and evaluation metrics using Lightning's built-in logging
  - Configures a simple Adam optimizer

Intended use:
  - Provide a clean, readable baseline training loop in Lightning
  - Separate "model architecture" from "training mechanics"
  - Demonstrate how to log multiple losses/metrics consistently

Key batch contract:
  - batch["ts"]       : torch.Tensor input stack (e.g., [B, C, T, H, W])
  - batch["forecast"] : torch.Tensor target values (e.g., [B] or [B,])

Optional preprocessing:
  - If ``preprocess_fn`` is provided to ``__init__``, it is called on the batch dict
    before every model call. This is the intended hook for input transformations (such
    as inverse-normalizing SDO channels) that should not live inside the model.

Key metrics contract (the `metrics` dict passed to __init__):
  - metrics["train_loss"]    : callable(output, target) -> (loss_dict, weight_list)
        Backpropagated. Logged as "train_loss".
  - metrics["val_loss"]      : callable(output, target) -> (loss_dict, weight_list)
        Optional. Logged as "val_loss" and therefore what ModelCheckpoint monitors.
        Falls back to metrics["train_loss"] when absent.
  - metrics["train_metrics"] : callable(output, target) -> (metric_dict, weight_list)
  - metrics["val_metrics"]   : callable(output, target) -> (metric_dict, weight_list)
        Reported only. These do NOT affect checkpoint selection — "val_loss" does.

Where:
  - loss_dict / metric_dict map string names -> torch scalar tensors
  - weight_list is a list-like of floats (or tensors) aligned with the dict iteration order
    used by this baseline to form a weighted sum loss.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import lightning as L
import torch


# Type aliases for clarity in documentation / teaching.
LossDict = Mapping[str, torch.Tensor]
MetricDict = Mapping[str, torch.Tensor]
Weights = Any  # often a list[float] or list[torch.Tensor]


class SimSiamLightningModule(L.LightningModule):
    """
    PyTorch LightningModule for simsearch training.

    This class wraps:
      (1) a user-provided PyTorch model (nn.Module-like) and
      (2) a set of loss/metric callables packaged in the `metrics` dictionary.

    Parameters
    ----------
    model:
        A callable model (typically torch.nn.Module) that accepts the batch input tensor
        `x = batch["ts"]` and returns predictions `output`.

    metrics:
        Dictionary containing the training loss function and metric functions.

        Required keys:
          - "train_loss": callable(output, target) -> (losses, weights)
              losses: dict[str, torch.Tensor] scalar losses
              weights: list-like aligned with iteration order of losses.keys()
          - "train_metrics": callable(output, target) -> (metrics, weights)
          - "val_metrics": callable(output, target) -> (metrics, weights)

        Optional key:
          - "val_loss": callable(output, target) -> (losses, weights)
              The validation objective. Defaults to "train_loss" when not supplied, so
              older metrics dicts keep working unchanged.

        The module uses:
          - train_loss in training_step, backpropagated and logged as "train_loss"
          - val_loss in validation_step, logged as "val_loss" — the quantity
            ModelCheckpoint monitors
          - train_metrics logged during training_step (if weights is non-empty)
          - val_metrics logged during validation_step (if weights is non-empty).
            Reported only; they do not influence checkpoint selection.

    lr:
        Learning rate for the Adam optimizer.

    batch_size:
        Optional batch size passed to Lightning's `self.log(..., batch_size=...)`.
        This improves correct averaging behavior when using distributed settings
        or variable batch sizes.

    preprocess_fn:
        Optional callable applied to the batch dict before every model call.
        Signature: ``(batch: dict) -> dict``. Use this to apply input
        transformations (e.g., ``destandardize_channels``) without
        embedding them in the model itself.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Callable[..., Tuple[Dict[str, torch.Tensor], Weights]]],
        lr: float,
        batch_size: Optional[int] = None,
        preprocess_fn: Optional[Callable[[Dict], Dict]] = None,
    ):
        super().__init__()
        self.automatic_optimization = True
        self.batch_size = batch_size
        self.model = model
        self.preprocess_fn = preprocess_fn

        self.lr = lr

    @staticmethod
    def _combine_losses(loss_dict: LossDict, weights: Weights) -> torch.Tensor:
        """Return a weighted sum of the losses in ``loss_dict``.

        ``weights`` must be aligned with ``loss_dict.keys()`` iteration order.
        Raises ``ValueError`` if ``loss_dict`` is empty.
        """
        loss = None
        for n, key in enumerate(loss_dict.keys()):
            component = loss_dict[key] * weights[n]
            loss = component if loss is None else (loss + component)
        if loss is None:
            raise ValueError("loss_dict is empty; cannot compute a scalar loss.")
        return loss

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass used by Lightning and by explicit calls in steps.

        Parameters
        ----------
        batch:
            Batch dict (at minimum contains ``"ts"`` and ``"ts_aug"``).

        Returns
        -------
        torch.Tensor
            Model predictions for the batch.
        """
        return self.model(x)
    

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Runs one training step on a single batch.

        Workflow
        --------
        1) Extract inputs and targets from the batch:
              x = batch["ts"]
              target = batch["forecast"]
        2) Compute model output:
              output = self(x)
        3) Compute per-component losses and combine via provided weights:
              training_losses, training_loss_weights = training_loss(output, target)
        4) Log:
              - total weighted loss as "train_loss" (progress bar)
              - each component loss as "train_loss_<name>"
              - training metrics as "train_metric_<name>" (if any)

        Notes
        -----
        - Targets are reshaped to shape [B, 1] by unsqueeze(1) to match a common
          "single output per sample" convention.
        - The loss combination depends on dict iteration order; ensure loss dict
          insertion order is consistent if that matters.

        Returns
        -------
        torch.Tensor
            The scalar training loss used for backpropagation.
        """
        batch1, batch2 = batch
        if self.preprocess_fn is not None:
            batch1 = self.preprocess_fn(batch1)
            batch2 = self.preprocess_fn(batch2)

        z0, p0 = self(batch1)
        z1, p1 = self(batch2)
        loss = -0.5 * (torch.nn.functional.cosine_similarity(p0, z1, dim=-1).mean() + 
                       torch.nn.functional.cosine_similarity( p1, z0, dim=-1).mean())

        with torch.no_grad():
            z0_norm = torch.nn.functional.normalize(z0, dim=-1)
            z1_norm = torch.nn.functional.normalize(z1, dim=-1)

            std = 0.5 * (
                z0_norm.std(dim=0, unbiased=False).mean()
                + z1_norm.std(dim=0, unbiased=False).mean()
            )
            collapse_level = (1.0 - z0.shape[-1] ** 0.5 * std).clamp(0, 1)

        self.log_dict(
            {
                "train_loss": loss,
                "collapse_level": collapse_level,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch1["ts"].shape[0],
            sync_dist=True,)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer used by Lightning.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer over all module parameters with learning rate `self.lr`.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
