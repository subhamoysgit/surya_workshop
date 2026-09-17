import numpy as np
import pandas as pd
from typing import Callable, Literal
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset
import cv2
from einops import rearrange

def perform_augmentation(ss, x, A):
    """Apply a centered rotation or reflection to an image stack and its bounding box.

    ### Parameters
    - `ss`: NumPy image stack with shape `(channels, height, width)`. Use
      multiple channels so OpenCV preserves the channel dimension.
    - `x`: Bounding-box coordinates in `(x0, x1, y0, y1)` order, where `x`
      denotes columns and `y` denotes rows.
    - `A`: NumPy affine matrix with shape `(2, 2)`, such as rotations or 
    reflections in `SimSearchSDataset.ADDICT`.

    ### Returns
    `(ss_t, bounds)` containing the transformed image stack with the
    same shape as `ss` and integer bounds `(x0_t, x1_t, y0_t, y1_t)`.
    """

    _, height, width = ss.shape
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    x_ = np.array([[x[0], x[1]],[x[2], x[3]]]) - center[0]
    x_t = center[0] + A@x_
    x0_t, y0_t = int(np.min(x_t[0,:])), int(np.min(x_t[1,:]))
    x1_t, y1_t = int(np.max(x_t[0,:])), int(np.max(x_t[1,:]))
    translation = center - A @ center
    M = np.column_stack((A, translation)).astype(np.float32)
    ss_t = cv2.warpAffine(rearrange(ss, 'c h w -> h w c'),
                       M, dsize=(width, height),
                       flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0,)
    return rearrange(ss_t, 'h w c -> c h w'), (x0_t, x1_t, y0_t, y1_t)

class SimSearchSDataset(HelioNetCDFDataset):
    """
    Template child class of HelioNetCDFDataset showing how to build a downstream dataset.
    Extends the base class with a flare intensity label aligned to the Surya index.

    All ``HelioNetCDFDataset`` keyword arguments (``index_path``, ``scalers``, ``channels``,
    ``s3_cache_dir``, etc.) are accepted via ``**kwargs`` and forwarded to the base class.
    ``load_forecast_frames`` defaults to ``False`` here (flare forecasting supplies its own
    labels, so future Surya frames are never fetched); pass it explicitly to override.

    Additional Args:
        return_surya_stack: If True (default), include the Surya image stack in the returned dict.
            Set to False to return only the flare intensity label (useful for label inspection).
        max_number_of_samples: Cap the dataset length at this value. Useful for quick experiments.
        label_transform: Optional callable applied to the ``intensity`` column of the flare index
            to produce the ``normalized_intensity`` label. Signature:
            ``(series: pd.Series) -> pd.Series``.  If ``None``, the raw intensity values are
            used as-is. Define this at the call site (e.g., in ``build_datasets()``) to keep
            normalization logic out of the dataset class.
        ds_flare_index_path: Path to the downstream flare intensity CSV index.
        ds_time_column: Column name in the flare index to use as the event timestamp.
        ds_time_tolerance: Maximum allowed time offset when matching Surya and DS indices
            (e.g., ``"15min"``). Unmatched entries are dropped.
        ds_match_direction: Merge direction passed to ``pd.merge_asof``. Use ``"forward"``
            for causal prediction (predict flares from prior solar state).

    Raises:
        ValueError: If ``ds_flare_index_path`` is not provided, or if no overlap exists
            between the Surya and DS indices within the specified tolerance.
    """
    ADDICT = {'90':(0, 1, -1, 0),
        '-90':(0, -1, 1, 0),
        '180':(-1, 0, 0, -1),
        'hflip':(-1, 0, 0, 1),
        '90+hflip':(0, 1, 1, 0),
        '-90+hflip':(0, -1, -1, 0),
        '180+hflip':(1, 0, 0, -1)}

    def __init__(
        self,
        ds_sim_index_path: str | None = None,
        # All HelioNetCDFDataset parameters (index_path, scalers, channels, s3_*, etc.)
        **kwargs,
    ):
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)

        # # Load ds index and find intersection with Surya index
        if ds_sim_index_path is not None:
            self.ds_index = pd.read_csv(ds_sim_index_path)
            self.ds_index["timestep"] = pd.to_datetime(self.ds_index["timestep"], utc=True)


    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:
                forecast (np.float32): Normalized log10 flare intensity label.
                ds_index (str): ISO-format timestamp from the flare index.
            When ``return_surya_stack=True``, also includes all keys from
            ``HelioNetCDFDataset.__getitem__`` (ts, time_delta_input, lead_time_delta, etc.).
        """
        sample = super().__getitem__(idx=idx)
        timestep = pd.to_datetime(self.valid_indices[idx], utc=True)
        matches = self.ds_index.loc[self.ds_index["timestep"].eq(timestep)]
        coords = matches[["x0", "x1", "y0", "y1"]].iloc[0].astype(int).to_list()
        ss = sample['ts']
        mask = np.zeros_like(ss[:1,:,:,:])
        mask[:,:,coords[2]:coords[3],coords[0]:coords[1]] = 1
        ss = np.concatenate((ss, mask), axis=0)
        key = np.random.choice(list(self.ADDICT.keys()))
        ss_t_0, coords_t = perform_augmentation(ss[:,0,:,:], coords, np.array(self.ADDICT[key]).reshape(2, 2)) #Xt=AX
        ss_t_1, _ = perform_augmentation(ss[:,1,:,:], coords, np.array(self.ADDICT[key]).reshape(2, 2))
        ss_t = np.stack((ss_t_0, ss_t_1), axis=1)
        
        dict_ = {'ts':ss,
                'bbox':coords,
                'time_delta_input': sample['time_delta_input'],
                'key': 'identity'}
        dict_aug = {'ts':ss_t,
                'bbox':coords_t,
                'time_delta_input': sample['time_delta_input'],
                'key': key}
        
        return dict_, dict_aug
        
        # return {'ts':ss, 
        #         'ts_aug': ss_t,
        #         'bbox':coords,
        #         'bbox_aug':coords_t, 
        #         'time_delta_input': sample['time_delta_input'],
        #         'key': key} 
                #'tstep': timestep}
