"""
check_sparsecl_training.py - does a SparseCL sparse model separate contradictions from confounders?

From embedding dumps of SparseCL's own test script (sparsecl_repo/dump_emb.py), for every test
query: the Hoyer sparsity of the difference between its sparse embedding and that of
  - its target (a contradiction),
  - its confounder: on vitaminc the sentence the query paraphrases; on SparseCL's datasets the
    query's highest-cosine passage that is not one of its contradictions (usually a paraphrase),
  - a random corpus passage,
and how often the sparse term alone ranks the target above the confounder. SparseCL's paper
(Figure 3) shows successful training: about 0.21 for everything before training, and about 0.63
for contradictions against 0.28 for paraphrases after.

    python check_sparsecl_training.py                                   # vitaminc, seeds 0-2
    python check_sparsecl_training.py --dataset hotpotqa \\
        --dumps results/val_emb_released.npz results/val_emb_ours.npz --out results/validate.json
"""
import argparse
import json
import math
import pickle

import numpy as np


def hoyer(x):
    n = math.sqrt(x.shape[-1])
    l1 = np.abs(x).sum(-1)
    l2 = np.linalg.norm(x, axis=-1)
    return (n - l1 / np.maximum(l2, 1e-12)) / (n - 1)


def confounders_by_cosine(z, qi, qs, qrels, pos, chunk=256):
    """the highest-cosine passage of each query that is neither relevant nor the query itself."""
    cd = z["cos_d"] / np.linalg.norm(z["cos_d"], axis=1, keepdims=True)
    cq = z["cos_q"][qi] / np.linalg.norm(z["cos_q"][qi], axis=1, keepdims=True)
    out = np.empty(len(qs), dtype=np.int64)
    for a in range(0, len(qs), chunk):
        S = cq[a:a + chunk] @ cd.T
        S[S > 0.9999] = -np.inf                              # the query's own text
        for j, q in enumerate(qs[a:a + chunk]):
            for p in qrels[q]:
                if p in pos:
                    S[j, pos[p]] = -np.inf
            if q in pos:
                S[j, pos[q]] = -np.inf
        out[a:a + chunk] = S.argmax(1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vitaminc")
    ap.add_argument("--dumps", nargs="+",
                    default=[f"results/sparsecl_emb_s{s}.npz" for s in (0, 1, 2)])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    name = (f"{args.dataset}_test_retrieval_final.pkl" if args.dataset == "arguana"
            else f"{args.dataset}_test_retrieval_gpt4_final.pkl")
    with open(f"{args.data}/{name}", "rb") as f:
        pickle.load(f)
        pickle.load(f)
        qrels = pickle.load(f)
    rng = np.random.default_rng(0)
    res = {}
    for path in args.dumps:
        z = np.load(path)
        pos = {str(p): i for i, p in enumerate(z["pids"])}
        qpos = {str(q): i for i, q in enumerate(z["qids"])}
        qs = [q for q in qrels if q in qpos and any(p in pos for p in qrels[q])]
        qi = np.array([qpos[q] for q in qs])
        ti = np.array([pos[next(p for p in qrels[q] if p in pos)] for q in qs])
        if args.dataset == "vitaminc":   # the symmetric benchmark: confounders by construction
            ci = np.array([pos[q.replace("-q", "-", 1)] for q in qs])
        else:
            ci = confounders_by_cosine(z, qi, qs, qrels, pos)
        ri = rng.integers(0, len(pos), len(qs))
        sq = z["sp_q"][qi]
        ht, hc, hr = (hoyer(z["sp_d"][i] - sq) for i in (ti, ci, ri))
        r = dict(target=float(np.median(ht)), confounder=float(np.median(hc)),
                 random=float(np.median(hr)), pairwise=float(np.mean(ht > hc)), queries=len(qs))
        res[path] = r
        print(f"{path}: median Hoyer  target {r['target']:.3f}  confounder {r['confounder']:.3f}  "
              f"random {r['random']:.3f}  | sparse term alone ranks the target above the confounder "
              f"for {100 * r['pairwise']:.1f}% of {len(qs)} queries")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
