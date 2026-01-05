import pandas as pd
from pathlib import Path
import re
def sample_head(csv_path, n=5):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    out_path = csv_path.with_name(csv_path.stem + "_sample" + csv_path.suffix)
    df.head(n).to_csv(out_path, index=False)
    return out_path

def extract_state(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)   
    out_path = csv_path.with_name(csv_path.stem + "_inits" + csv_path.suffix)
    state_pattern = r"\*\*Current State\*\*:\s*(.*?)\.\s*\*\*Task\*\*:"
    df["output"] = df["input"].str.extract(state_pattern, flags=re.DOTALL)[0]

    # remove the whole Current State line
    remove_pattern = r"\n?\*\*Current State\*\*:\s*.*?\.\s*(?=\*\*Task\*\*:)"
    df["input"] = df["input"].str.replace(remove_pattern, "", regex=True)

    # append header at the end
    df["input"] = df["input"].str.rstrip() + "\n**Current State**:"
    df.to_csv(out_path,index=False)
extract_state("pathway_cot_dataset_final.csv")
sample_head("pathway_cot_dataset_final_inits.csv")