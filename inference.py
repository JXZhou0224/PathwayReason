import numpy as np
import torch
import pandas as pd

from model.modelling_vae import VAEModel,VAEConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from train_vae import load_ckpt,load_config

import argparse

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

@torch.no_grad()
def infer_and_dump_npz(
    model:VAEModel,
    data_path,
    out_path,
    batch_size=4,
    num_samples=10000,
    device="cuda"
):
    model.eval()
    model.to(device)

    dataset = PathwaySentence(data_path,num_samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    zs = []

    for batch in tqdm(loader):
        mu, logvar = model.encode_text(batch)
        z = model.vae.reparametrize(mu, logvar)
        zs.append(z.detach().cpu())

    Z = torch.cat(zs, dim=0).numpy()  # (N, D)
    print(Z.shape)
    np.savez(out_path, Z=Z)

    return Z

@torch.no_grad()
def decode_text(
    model:VAEModel,
    embeds,
    device="cuda"
):
    model.eval()
    model.to(device)

    texts = model.decode_text(embeds)

    return texts

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default="runs/case_1_kl_weight_1/config.yaml")
    parser.add_argument("--ckpt_path", default="runs/case_1_kl_weight_1/ckpt/step_2600.pt")
    parser.add_argument("--data_path", default="data/pathway_cot_dataset_final_inits.csv")
    parser.add_argument("--out_path", default="data/embeds_intermediate.npz")

    args = parser.parse_args()

    cfg_path = args.cfg_path
    ckpt_path = args.ckpt_path
    data_path = args.data_path
    out_path = args.out_path
    cfg = load_config(cfg_path)
    vae_cfg = VAEConfig(**cfg["VAEConfig"])
    model = VAEModel(vae_cfg)
    model.load_ckpt(ckpt_path)
    infer_and_dump_npz(model,data_path,out_path,batch_size=80,num_samples=-1)
    # data = np.load(out_path)["Z"][:2]
    # data = torch.from_numpy(data).to("cuda")
    # print("decoding start")
    # texts = decode_text(model,data)
    # print(texts)