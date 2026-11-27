"""download.py - cache every model and dataset the experiments need (run on a machine with internet
access; setup.sh calls it). With --sparsecl_data, also SparseCL's original benchmarks into ./data."""
import argparse

from transformers import AutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--sparsecl_data", action="store_true")
args = ap.parse_args()

MODELS = ["BAAI/bge-base-en-v1.5", "SparseCL/BGE-SparseCL-arguana",
          "SparseCL/BGE-SparseCL-hotpotqa", "SparseCL/BGE-SparseCL-msmarco"]
MT = ["Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"]

for name in MODELS:
    AutoModel.from_pretrained(name)
    AutoTokenizer.from_pretrained(name)
    print("cached", name)
for name in MT:
    MarianMTModel.from_pretrained(name)
    MarianTokenizer.from_pretrained(name)
    print("cached", name)

from datasets import load_dataset
for split in ("train", "validation", "test"):
    print("cached tals/vitaminc", split, len(load_dataset("tals/vitaminc", split=split)), "rows")

if args.sparsecl_data:
    from huggingface_hub import snapshot_download
    snapshot_download("SparseCL/all-train-dev-test-data", local_dir="data")
    print("downloaded SparseCL's original data into ./data")
