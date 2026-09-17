"""
A simple linear regression model to be used as a baseline for flare forecasting.
"""

import torch
import torch.nn as nn
from einops import rearrange
import sys
# Append base path.  May need to be modified if the folder structure changes.
# It gives the notebook access to the wokshop_infrastructure folder.
import torch.nn.functional as F
import yaml


def destandardize_channels(batch: dict, channel_order: list, scalers: dict) -> dict:
    """Return a new batch dict with 'ts' moved from normalized space to signum-log space.

    This undoes the per-channel z-score ONLY. The signum-log compression applied by the
    dataset is deliberately left in place, so the result is
    ``sign(x*s) * log1p(|x*s|)`` — not raw DN/Gauss. Values spanning many orders of
    magnitude make poor features for a single linear layer, so log space is what the
    baseline wants.

    If you need true physical units (plotting, a physical-space loss), use
    ``HelioNetCDFDataset.inverse_transform_data()`` instead, which undoes both stages.
    See the "THE THREE SPACES" block in ``workshop_infrastructure/datasets/helio.py``.

    Args:
        batch: Batch dict containing at minimum a 'ts' key with shape (B, C, T, H, W).
        channel_order: Channel names in the same order as the C dimension of 'ts'.
        scalers: Dict mapping channel name -> scaler with an inverse_transform method.

    Returns:
        A new batch dict with 'ts' replaced by the de-standardized (signum-log) tensor.
    """
    x = batch["ts"].clone()
    with torch.no_grad():
        for i, channel in enumerate(channel_order):
            x[:, i, ...] = scalers[channel].inverse_transform(x[:, i, ...])
    return {**batch, "ts": x}


class SimSuryaModel(nn.Module):
    def __init__(self, model, bbox=False):
        """
        Initializes the RegressionFlareModel.

        Args:
            input_dim (int): The size of the input vector after channel and time dimensions are flattened.

        Note:
            This model expects 'ts' in the batch dict to already be in **signum-log** space
            (channel z-scores undone, log compression retained). Use
            destandardize_channels() to pre-process normalized SDO inputs before passing
            them here (e.g., via the preprocess_fn argument of FlareLightningModule).
        """
        super().__init__()
        self.model = model
        self.bbox = bbox
    def forward(self, x: dict) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (dict): Batch dict with 'ts' of shape (B, C, T, H, W) in signum-log space.

        B - Batch size
        C - Channels
        T - Time steps
        H - Height
        W - Width
        """

        height, width = x['ts'].shape[-2:]
        if height % 16 or width % 16:
            raise ValueError('Image dimensions must be divisible by patch_size')
        gy, gx = height // 16, width // 16
        x0, x1, y0, y1 = x['bbox']
        tx0, ty0 = x0 // 16, y0 // 16
        tx1, ty1 = (x1 + 16 - 1) // 16, (y1 + 16 - 1) // 16
        
        x0_, x1_, y0_, y1_ = x['bbox_aug']
        tx0_, ty0_ = x0_ // 16, y0_ // 16
        tx1_, ty1_ = (x1_ + 16 - 1) // 16, (y1_ + 16 - 1) // 16
        
        mask = x['ts'][:, -1, 0, :, :]
        ws = F.avg_pool2d(mask[:, None].float(),
                          kernel_size=16,
                          stride=16,)[:, 0]  # (B, gy, gx)
        ws = (ws > 0.5).to(torch.float32)
        ws = ws.unsqueeze(-1)  
        
        mask_aug = x['ts_aug'][:, -1, 0, :, :]
        ws_aug = F.avg_pool2d(mask_aug[:, None].float(),
                              kernel_size=16,
                              stride=16,)[:, 0]  # (B, gy, gx)
        ws_aug = (ws_aug > 0.5).to(torch.float32)
        ws_aug = ws_aug.unsqueeze(-1)  
        
        with torch.no_grad():
            token = self.model.embedding(x['ts'][:,:-1,...], x['time_delta_input'])
            token = self.model.backbone(token)
            token_aug = self.model.embedding(x['ts_aug'][:,:-1,...], x['time_delta_input'])
            token_aug = self.model.backbone(token_aug)
            grid = token.reshape(-1, gy, gx, token.shape[-1])
            grid_aug = token_aug.reshape(-1, gy, gx, token_aug.shape[-1])
            if self.bbox:
                grid = torch.mean(grid[:,ty0:ty1, tx0:tx1,:], axis=(1,2))
                grid_aug = torch.mean(grid_aug[:,ty0_:ty1_, tx0_:tx1_,:], axis=(1,2))
                return grid, grid_aug
            
        # (B, gy, gx, 1)
            grid = (
                (grid * ws).sum(dim=(1, 2))
                / ws.sum(dim=(1, 2)).clamp_min(1)
            )  
            
            grid_aug = (
                (grid_aug * ws_aug).sum(dim=(1, 2))
                / ws_aug.sum(dim=(1, 2)).clamp_min(1)
            )  
            return grid, grid_aug