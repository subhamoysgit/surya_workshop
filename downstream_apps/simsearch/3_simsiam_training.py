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
from downstream_apps.simsearch.models.simsearch_simsiam import SimSuryaModel
from downstream_apps.simsearch.lightning_modules.pl_simsearch_simsiam import SimSiamLightningModule
from downstream_apps.simsearch.metrics.simsearch_metrics import SimsearchMetrics
from dataclasses import asdict
from workshop_infrastructure.utils import (load_pretrained_weights, apply_peft_lora,)
torch.set_float32_matmul_precision('medium')

cfg = load_flare_config("./configs/config_script_beastie.yaml")
ensure_assets(cfg, which=["scalers", "weights"])
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

from torch.utils.data import Subset

subset_size = 10

indices = torch.randperm(
    len(train_dataset),
    generator=torch.Generator().manual_seed(42),
)[:subset_size].tolist()

train_subset = Subset(train_dataset, indices)

try:
    config = yaml.safe_load(open('./configs/config_script_beastie.yaml', "r"))
    print("Configuration loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure config_script.yaml exists in your current directory")
    raise

# build_scalers accepts a file path directly
scalers = build_scalers(info=config["data"]["scalers_path"])
L.seed_everything(42, workers=True)

m = cfg.model
basemodel = HelioSpectFormer(
    img_size=m.img_size,
    patch_size=m.patch_size,
    in_chans=m.in_channels,
    embed_dim=m.embed_dim,
    time_embedding=asdict(m.time_embedding),
    depth=m.depth,
    n_spectral_blocks=m.spectral_blocks,
    num_heads=m.num_heads,
    mlp_ratio=m.mlp_ratio,
    drop_rate=m.drop_rate,
    window_size=m.window_size,
    dp_rank=m.dp_rank,
    learned_flow=m.learned_flow,
    use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
    init_weights=m.init_weights,
    checkpoint_layers=m.checkpoint_layers,
    rpe=m.rpe,
    ensemble=m.ensemble,
    finetune=True,
    nglo=0,
    dtype=cfg.dtype,
)
load_pretrained_weights(basemodel, cfg.model.pretrained_path)
proj_hidden_dim = 512
pred_hidden_dim = 128
out_dim = 256

model = SimSuryaModel(
    basemodel,
    num_ftrs=m.embed_dim,
    proj_hidden_dim=proj_hidden_dim,
    pred_hidden_dim=pred_hidden_dim,
    out_dim=out_dim,
)

if cfg.model.freeze_backbone:
    basemodel.requires_grad_(False)
    
if cfg.model.use_lora:
    model = apply_peft_lora(model, cfg.model.lora_config)
    


train_loss_metrics = SimsearchMetrics("train_loss")

metrics = {'train_loss': train_loss_metrics}

lit_model = SimSiamLightningModule(model, metrics, 
                                     lr=cfg.learning_rate,
                                     batch_size=cfg.batch_size)
project_name = cfg.wandb_project

regime = (
    f"lora{m.lora_config.r}" if m.use_lora
    else "frozen" if m.freeze_backbone
    else "full"
)

run_name = (
    f"simsiam_proj{proj_hidden_dim}_pred{pred_hidden_dim}"
    f"_out{out_dim}_{regime}_bs{cfg.batch_size}"
    f"_n{len(train_subset)}"
)


#run_name = f"simsiam_surya_finetuning_bs{cfg.batch_size}_use_lora_{cfg.model.use_lora}_proj_dim"  # give your run a descriptive name

wandb_logger = WandbLogger(
    entity=cfg.wandb_entity,  # set wandb_entity in the config; null = personal account
    project=project_name,
    name=run_name,
    log_model=False,
    save_dir="./wandb/wandb_tmp",
)

csv_logger = CSVLogger("runs", name=project_name)

max_epochs = cfg.max_epochs
checkpoint_callback = ModelCheckpoint(
    dirpath=cfg.output.ckpt_dir,
    filename="best",
    monitor="train_loss_epoch",
    mode="min",
    save_top_k=1,
    save_last=True,
)
# -------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------
trainer = L.Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    devices="auto",
    logger=[wandb_logger, csv_logger],
    log_every_n_steps=2,
    callbacks=[checkpoint_callback],
)

train_data_loader = DataLoader(
    dataset = train_subset,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

trainer.fit(lit_model, train_data_loader)


