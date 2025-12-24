import argparse
import yaml
from dataclasses import asdict
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.modelling_vae import VAEModel
from model.configs import VAEConfig, TrainConfig

# ======================================================
# utils
# ======================================================
def load_config(path):
    path = Path(path)
    if path.suffix in [".yml", ".yaml"]:
        return yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        return json.loads(path.read_text())
    else:
        raise ValueError("Unsupported config format")


def dump_args(args, path):
    yaml.safe_dump(vars(args), open(path, "w"), sort_keys=False)


def make_run_dir(base="runs", name=None):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{ts}_{name}" if name else ts
    run_dir = Path(base) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_ckpt(path, model:VAEModel, optimizer, step):
    torch.save(
        {
            "model": model.vae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_ckpt(path: Path, model: VAEModel, optimizer:torch.optim.Optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.vae.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)




def param_count(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


# ======================================================
# dataset (placeholder)
# ======================================================
class PathwaySentence(Dataset):
    def __init__(self, data_path="data/pathway_cot_dataset_final.csv",num_samples=10000):
        if num_samples == -1:
            num_samples = None
        self.data = pd.read_csv(data_path,nrows=num_samples)
        self.data["system"] = (
            self.data["instruction"].fillna("") +
            self.data["input"].fillna("")
        )
        self.data.rename(columns={"output": "content"}, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "system": self.data.iloc[idx]["system"],
            "content": self.data.iloc[idx]["content"],
        }

# ======================================================
# training step
# ======================================================
def train_step(model:VAEModel, batch, optimizer:torch.optim.Optimizer, kl_weight:float):
    model.train()
    recon, kl = model(batch)

    loss = recon + kl_weight * kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.item(),
        "recon": recon.item(),
        "kl": kl.item(),
    }

def validate(model:VAEModel):
    pass

# ======================================================
# main
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    vae_cfg = VAEConfig(**cfg["VAEConfig"])
    train_cfg = TrainConfig(**cfg["TrainConfig"])
    torch.manual_seed(cfg.get("seed", 42))
    print(vae_cfg)
    print(train_cfg)
    # ------------------
    # run directory
    # ------------------
    run_dir = make_run_dir(name=args.exp_name)
    dump_args(args, run_dir / "args.yaml")
    yaml.safe_dump(asdict(vae_cfg), open(run_dir/"vae_cfg.yaml","w"), sort_keys=False)
    yaml.safe_dump(asdict(train_cfg), open(run_dir/"train_cfg.yaml","w"), sort_keys=False)
    ckpt_dir = run_dir / "checkpoints"

    # ------------------
    # model
    # ------------------
    model = VAEModel(vae_cfg).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **train_cfg.optim_cfg
    )
    print("model_params:")
    print(param_count(model))
    step = 0
    if args.resume:
        ckpt_path = args.ckpt or ckpt_dir / "latest.pt"
        step = load_ckpt(ckpt_path, model, optimizer)

    # ------------------
    # data (replace later)
    # ------------------
    dataset = PathwaySentence(data_path=train_cfg.data_path,num_samples=train_cfg.num_samples)
    print(len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False
    )

    # ------------------
    # training loop
    # ------------------
    for epoch in range(train_cfg.epochs):
        for batch in loader:
            if(step < train_cfg.warmup_steps):
                kl_weight = step/train_cfg.warmup_steps * train_cfg.kl_weight
            else:
                kl_weight = train_cfg.kl_weight
            if step % 100 == 0:
                torch.cuda.empty_cache()

            metrics = train_step(
                model,
                batch,
                optimizer,
                kl_weight=kl_weight,
            )
            step += 1

            if step % train_cfg.log_every == 0:
                print(f"[step {step}] {metrics}")
                print(f"[step {step}] Max mem: ",torch.cuda.max_memory_allocated() / 1e9, "GB")

            if step % train_cfg.save_every == 0:
                save_ckpt(
                    ckpt_dir / f"step_{step}.pt",
                    model,
                    optimizer,
                    step,
                )
                save_ckpt(
                    ckpt_dir / "latest.pt",
                    model,
                    optimizer,
                    step,
                )


if __name__ == "__main__":
    main()
