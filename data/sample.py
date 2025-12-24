import pandas as pd
from pathlib import Path

def sample_head(csv_path, n=5):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    out_path = csv_path.with_name(csv_path.stem + "_sample" + csv_path.suffix)
    df.head(n).to_csv(out_path, index=False)
    return out_path

sample_head("pathway_cot_dataset_final.csv")