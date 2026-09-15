import sys
sys.path.append("/home/bjha/researcd/projects/suryaworksop/surya_workshop/")

from pathlib import Path
import yaml
import torch
import warnings
from workshop_infrastructure.models.helio_spectformer import HelioSpectFormer
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset
from workshop_infrastructure.utils import build_scalers

warnings.filterwarnings("ignore", category=FutureWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

config_path = "config_script.yaml"
config = yaml.safe_load(open(config_path, "r"))
print("Configuration loaded successfully!")
scalers = build_scalers(info=config["data"]["scalers_path"])

dataset = HelioNetCDFDataset(
    index_path=config["data"]["train_data_path"],
    time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
    time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
    n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
    rollout_steps=config["training"]["rollout_steps"],
    channels=config["data"]["channels"],
    drop_hmi_probability=config["training"]["drop_hmi_probability"],
    use_latitude_in_learned_flow=config["training"]["use_latitude_in_learned_flow"],
    scalers=scalers,
    phase="train",
    s3_mode=config["data"]["s3_mode"],
    s3_storage_options={"anon": config["data"]["s3_anon"]},
    s3_cache_dir=config["data"]["s3_cache_dir"],
    load_forecast_frames=False,
)

model = HelioSpectFormer(
    #### Backbone (Surya 366M defaults — must match the pretrained checkpoint)
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
    window_size=config["model"]["window_size"],
    dp_rank=config["model"]["dp_rank"],
    learned_flow=config["model"]["learned_flow"],
    use_latitude_in_learned_flow=config["training"]["use_latitude_in_learned_flow"],
    init_weights=config["model"]["init_weights"],
    checkpoint_layers=config["model"]["checkpoint_layers"],
    rpe=config["model"]["rpe"],
    ensemble=config["model"]["ensemble"],
    nglo=0,#config["model"]["nglo"],
    finetune=True,
)
print("Model intitialized successfully!")

inputdata = dataset[0]

input_x = {
    "ts": torch.as_tensor(
        inputdata["ts"], dtype=torch.float32, device=device
    ).unsqueeze(0),
    "time_delta_input": torch.as_tensor(
        inputdata["time_delta_input"], dtype=torch.float32, device=device
    ).unsqueeze(0),
}


checkpoint_state=torch.load(config["model"]["pretrained_path"],
                           weights_only=True,
                           map_location="cpu")
print("Checkpoint loaded successfully!")

encoder_state = {
    name: value
    for name, value in checkpoint_state.items()
    if not name.startswith("unembed.")
}
model.load_state_dict(encoder_state, strict=True)
_ = model.to(device)
_ = model.eval()

with torch.inference_mode():
    patch_embeddings = model(input_x)          # (B, 65536, 1280)
    global_embeddings = patch_embeddings.mean(dim=1)  # (B, 1280)
print(patch_embeddings.shape)
print(global_embeddings.shape)
