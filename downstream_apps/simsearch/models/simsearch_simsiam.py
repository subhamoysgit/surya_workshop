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
    def __init__(self, model,num_ftrs=1280,
                 proj_hidden_dim=640,
                 pred_hidden_dim=640,
                 out_dim=1280,):
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
        self.backbone = model

        self.head_projection = nn.Sequential(
            nn.Linear(num_ftrs, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),

            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),

            nn.Linear(proj_hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

        self.head_prediction = nn.Sequential(
            nn.Linear(out_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),

            nn.Linear(pred_hidden_dim, out_dim),
        )
    
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
        
        mask = x['ts'][:, -1, 0, :, :]
        ws = F.avg_pool2d(mask[:, None].float(),
                          kernel_size=16,
                          stride=16,)[:, 0]  # (B, gy, gx)
        ws = (ws > 0.5).to(torch.float32)
        ws = ws.unsqueeze(-1)  
        token = self.backbone.embedding(x['ts'][:,:-1,...], x['time_delta_input'])
        token = self.backbone.backbone(token)
        grid = token.reshape(-1, gy, gx, token.shape[-1])
        
       # (B, gy, gx, 1)
        grid = (
            (grid * ws).sum(dim=(1, 2))
            / ws.sum(dim=(1, 2)).clamp_min(1)
        )  

        f = grid.flatten(start_dim=1)
        z = self.head_projection(f)
        p = self.head_prediction(z)
        z = z.detach()
        
        return z, p