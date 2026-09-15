import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import xarray
from urllib.parse import urlparse
from astropy.io import fits
from sunpy.map import Map
from sunpy.visualization import colormaps as cm
import pandas as pd


def plot_surya_crop(image_2d, bbox, other_image, cmap="hmimag", title="HMI magnetogram"):
    """Plot a 2D image with crop region marked, the cropped region, and a given comparison image (already crop-sized)."""

    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    image_2d = to_numpy(image_2d)
    other_image = to_numpy(other_image)
    x0, x1, y0, y1 = bbox
    crop = image_2d[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.01, top=0.9)

    vmin, vmax = -1500, 1500 #np.nanpercentile(image_2d, [1, 99])
    axes[0].imshow(image_2d, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    axes[0].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         linewidth=1.5, edgecolor="lime", facecolor="none"))

    axes[1].imshow(crop, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

    axes[2].imshow(other_image, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(r"SURYA Stack HMI $B_z$")
    axes[1].set_title(f"Cropped $B_z$ (from SURYA Stack)")
    axes[2].set_title("Original SHARP")
    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes

output_csv_path = Path("/d0/bjha/data/surya/workshop2026/indices/simsearch_surya_sharp_indices.csv")
datapath = Path("/d0/subhamoy/sharps/surya_embeddings")
surya_data_path = Path("/d0/bjha/data/surya/workshop2026/surya_ws_cache")

plot_path = Path("/d0/bjha/data/surya/workshop2026/sharp_plots")
plot_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(output_csv_path)
files = list(sorted(df['embed_path'].unique()))

for i, file in enumerate(files):
    print(f"Processing file {i+1} {Path(file).stem} out of {len(files)}.")
    try:
        ds = xarray.load_dataset(file)
        coords = ds.attrs["x0"], ds.attrs["x1"], ds.attrs["y0"], ds.attrs["y1"]
        
        surya_file = surya_data_path/urlparse(ds.attrs["surya_file"]).path.lstrip("/")
        cart_file = ds.attrs["cart_file"]
        with fits.open(cart_file) as hdul:
            data_sharp = hdul[1].data
            hdr = hdul[1].header
        data_sharp = Map(data_sharp, hdr).rotate().data
        bz = xarray.load_dataset(surya_file)["hmi_bz"]    
        ds.close()
        title = f"Harp No: {ds.attrs['harpnum']}   Reference Time: {ds.attrs['reference_time']}"
        fig, ax = plot_surya_crop(bz, coords, data_sharp, title=title)
        fig.savefig(plot_path/(Path(file).stem+".png"))
        plt.close()
    except Exception as e:
        print(e)
        continue
   