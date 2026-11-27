"""
restore_triples.py - regenerates the benchmark's training-triple files
(data/vitaminc_{train,dev}_gpt4_final.csv) from its retrieval files
(data/vitaminc_{train,dev}_retrieval_gpt4_final.pkl), exactly as build_vitaminc.py writes them.

For pair i of a split, the corpus holds sentences a = "{split}-{i}-a" and b = "{split}-{i}-b", and
the queries hold their paraphrases "{split}-q{i}-a" and "{split}-q{i}-b". Each pair contributes the
triples (paraphrase of a, b, a) and (paraphrase of b, a, b), as (query, contradiction, confounder).

    python restore_triples.py --data data
"""
import argparse
import csv
import os
import pickle
import re

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default="data")
args = ap.parse_args()

for split in ("train", "dev"):
    with open(os.path.join(args.data, f"vitaminc_{split}_retrieval_gpt4_final.pkl"), "rb") as f:
        corpus, queries, qrels = pickle.load(f), pickle.load(f), pickle.load(f)
    ids = sorted({int(m.group(1)) for k in corpus for m in [re.fullmatch(rf"{split}-(\d+)-[ab]", k)] if m})
    assert ids == list(range(len(ids))), f"{split}: pair ids are not 0..n-1"
    triples = []
    for i in ids:
        a, b = corpus[f"{split}-{i}-a"]["text"], corpus[f"{split}-{i}-b"]["text"]
        qa, qb = queries[f"{split}-q{i}-a"], queries[f"{split}-q{i}-b"]
        assert qrels[f"{split}-q{i}-a"] == {f"{split}-{i}-b": 1} and qrels[f"{split}-q{i}-b"] == {f"{split}-{i}-a": 1}
        triples += [(qa, b, a), (qb, a, b)]                 # (query, contradiction, confounder)
    out = os.path.join(args.data, f"vitaminc_{split}_gpt4_final.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sent0", "sent1", "hard_neg"])
        w.writerows(triples)
    print(f"{split}: {len(ids):,} pairs -> {len(triples):,} triples written to {out}")
