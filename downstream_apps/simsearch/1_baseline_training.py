import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from torch.utils.data import DataLoader

import torch
import yaml

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

# Append base path.  May need to be modified if the folder structure changes.
# It gives the notebook access to the wokshop_infrastructure folder.
sys.path.append("../../")
 
# Append Surya path. May need to be modified if the folder structure changes.
# It gives the notebook access to surya's release code.

from workshop_infrastructure.utils import build_scalers  # Data scaling utilities for Surya stacks
from downstream_apps.template.configs import load_flare_config
from workshop_infrastructure.assets import ensure_assets
from downstream_apps.simsearch.datasets.simsearch_dataset import SimSearchSDataset
from workshop_infrastructure.models.helio_spectformer import HelioSpectFormer
from downstream_apps.simsearch.models.simsearch_baseline import SimSuryaModel
from downstream_apps.simsearch.lightning_modules.pl_simsearch_baseline import SimsearchLightningModule
from downstream_apps.simsearch.metrics.simsearch_metrics import SimsearchMetrics
torch.set_float32_matmul_precision('medium')

cfg = load_flare_config("./configs/config_script_beastie.yaml")
ensure_assets(cfg, which=["scalers"])
# Now that scalers.yaml is guaranteed to be on disk, load it. build_scalers()
# accepts the resolved path directly.
scalers = build_scalers(info=cfg.data.scalers_path)

train_dataset = SimSearchSDataset(
    #### All these lines are required by the parent HelioNetCDFDataset class
    index_path=cfg.data.valid_data_path,
    time_delta_input_minutes=cfg.data.time_delta_input_minutes,
    time_delta_target_minutes=cfg.data.time_delta_target_minutes,
    n_input_timestamps=cfg.model.time_embedding.time_dim,
    rollout_steps=cfg.rollout_steps,
    channels=cfg.data.channels,
    drop_hmi_probability=cfg.drop_hmi_probability,
    use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
    scalers=scalers,
    phase="valid",          # "val" disables random channel masking and flips
    # How s3:// paths in the index are read: "download" (default) | "simplecache" | "stream".
    # "download" fetches each file into s3_cache_dir first — recommended for NetCDF.
    s3_mode=cfg.data.s3_mode,
    s3_storage_options={"anon": cfg.data.s3_anon},
    s3_cache_dir=cfg.data.s3_cache_dir,
    #### Put your downstream (DS) specific parameters below this line
    ds_sim_index_path=cfg.data.flare_index_path,
)


try:
    config = yaml.safe_load(open('./configs/config_script_beastie.yaml', "r"))
    print("Configuration loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure config_script.yaml exists in your current directory")
    raise

# build_scalers accepts a file path directly
scalers = build_scalers(info=config["data"]["scalers_path"])


basemodel = HelioSpectFormer(
        #### Surya 366M defaults — must match the pretrained checkpoint
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=config["model"]["in_channels"],
        embed_dim=config["model"]["embed_dim"],
        time_embedding=config["model"]["time_embedding"],
        depth=config["model"]["depth"],
        n_spectral_blocks=config["model"]["spectral_blocks"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        drop_rate=config["model"]["drop_rate"],
        dtype=torch.float32,
        window_size=config["model"]["window_size"],
        dp_rank=config["model"]["dp_rank"],
        learned_flow=config["model"]["learned_flow"],
        use_latitude_in_learned_flow=config["training"]["use_latitude_in_learned_flow"],
        init_weights=config["model"]["init_weights"],
        checkpoint_layers=config["model"]["checkpoint_layers"],
        rpe=config["model"]["rpe"],
        ensemble=config["model"]["ensemble"],
        nglo=config["model"]["nglo"],
        finetune=False,
    )
model = SimSuryaModel(basemodel)
L.seed_everything(42, workers=True)

train_loss_metrics = SimsearchMetrics("train_loss")

metrics = {'train_loss': train_loss_metrics}

lit_model = SimsearchLightningModule(model, metrics, 
                                     lr=cfg.learning_rate,
                                     batch_size=2)
project_name = cfg.wandb_project
run_name = "baseline_simsearch_bs2"  # give your run a descriptive name

wandb_logger = WandbLogger(
    entity=cfg.wandb_entity,  # set wandb_entity in the config; null = personal account
    project=project_name,
    name=run_name,
    log_model=False,
    save_dir="./wandb/wandb_tmp",
)

csv_logger = CSVLogger("runs", name=project_name)

max_epochs = 2

# -------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------
trainer = L.Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    devices="auto",
    logger=[wandb_logger, csv_logger],
    log_every_n_steps=2,
)

train_data_loader = DataLoader(
                dataset=train_dataset,
                batch_size=2,
                num_workers=0
            )
trainer.fit(lit_model, train_data_loader)


