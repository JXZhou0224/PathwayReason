from dataclasses import dataclass, asdict, field
import yaml



@dataclass
class VAEConfig:
    lm_model_name: str = "Qwen/Qwen2.5-0.5B"
    latent_token_n: int = 4
    dim_latent: int = 128
    dim_ae: int = 128
    depth: int = 3
    dim_head: int = 12
    latent_seed: int = 42
    max_output_token: int = 512
    device: str = "cuda"  # or "cpu"

@dataclass
class TrainConfig:
    data_path: str = "data/pathway_cot_dataset_final.csv"
    num_samples: int = -1
    batch_size: int = 8
    epochs: int = 10
    kl_weight: float = 0.1
    log_every: int = 10
    save_every: int = 500
    warmup_steps: float =  1000
    optim_cfg: dict = field(default_factory=lambda:{
        "lr": 3e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999]
    })
if __name__ == "__main__":
    yaml.safe_dump(asdict(VAEConfig()), open("vae_config.yaml", "w"))
    yaml.safe_dump(asdict(TrainConfig()), open("train_config.yaml", "w"))