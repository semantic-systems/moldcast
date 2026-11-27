"""
diagnose_sparsecl.py - the two diagnostics of diagnose_ours.py, for SparseCL trained on VitaminC:

  pairwise    share of test queries whose target scores above its confounder
  test-only   NDCG@10 against the test split's own passages only, by exact scoring

computed from the embeddings that SparseCL's own test script produces (sparsecl_repo/dump_emb.py,
see make_dump_script.py), with SparseCL's scoring rule, as in its dist() and process_query():
    score(q, d) = cos(q, d) + alpha * Hoyer(sparse(d) - sparse(q)),
    Hoyer(x) = (sqrt(n) - |x|_1 / |x|_2) / (sqrt(n) - 1), and -1e9 if |x|_2 < 1e-3,
alpha being the one their dev search chose. Query-free: every query's sparse embedding replaced by
the average one (as sparsecl_repo/test_qf.py does).

    python diagnose_sparsecl.py --data data --seeds 0 1 2 --out results/diagnose_sparsecl.json
"""
import argparse
import json
import math
import os
import pickle
import re

import numpy as np
import torch


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dumps", default="results/sparsecl_emb_s{seed}.npz")
    ap.add_argument("--dataset", default="vitaminc", help="the benchmark (vitaminc)")
    ap.add_argument("--outputs", default="sparsecl_repo/test_results/test_{dataset}_bge_sparsity_s{seed}"
                                         "/{dataset}_parallel_output")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", default="results/diagnose_sparsecl.json")
    return ap.parse_args()


def their_alpha(path):
    for line in open(path):
        if line.startswith("final alpha"):
            return float(line.split()[2])
    raise SystemExit(f"no 'final alpha' in {path}")


def hoyer(diff, sqrt_dim):
    """their dist(..., 'hoyer') for a batch of difference vectors (last dimension)."""
    l1 = diff.abs().sum(-1)
    l2 = diff.norm(dim=-1)
    h = (sqrt_dim - l1 / l2.clamp_min(1e-12)) / (sqrt_dim - 1)
    return torch.where(l2 < 1e-3, torch.full_like(h, -1e9), h)


def ndcg10(S, target_cols):
    top = S.topk(10, dim=1).indices
    hit = top == target_cols[:, None]
    rank = hit.float().argmax(1) + 1
    return float(torch.where(hit.any(1), 1.0 / torch.log2(rank.float() + 1),
                             torch.zeros_like(rank, dtype=torch.float)).mean())


def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(args.data, f"{args.dataset}_test_retrieval_gpt4_final.pkl"), "rb") as f:
        pickle.load(f)
        pickle.load(f)
        qrels = pickle.load(f)
    res = {}
    for s in args.seeds:
        z = np.load(args.dumps.format(seed=s))
        pids, qids = [str(p) for p in z["pids"]], [str(q) for q in z["qids"]]
        pos = {p: i for i, p in enumerate(pids)}
        qpos = {q: i for i, q in enumerate(qids)}
        test_qids = [q for q in qrels if q in qpos]
        qi = torch.tensor([qpos[q] for q in test_qids])
        ti = torch.tensor([pos[next(iter(qrels[q]))] for q in test_qids])
        ci = torch.tensor([pos[q.replace("-q", "-", 1)] for q in test_qids])
        test_cols = [i for i, p in enumerate(pids) if p.startswith("test-")]
        col_in_test = {c: j for j, c in enumerate(test_cols)}
        tgt_test = torch.tensor([col_in_test[int(t)] for t in ti]).to(dev)

        cd = torch.tensor(z["cos_d"], dtype=torch.float32, device=dev)
        cq = torch.tensor(z["cos_q"], dtype=torch.float32, device=dev)
        cd, cq = cd / cd.norm(dim=1, keepdim=True), cq / cq.norm(dim=1, keepdim=True)
        sd = torch.tensor(z["sp_d"], dtype=torch.float32, device=dev)
        sq_all = torch.tensor(z["sp_q"], dtype=torch.float32, device=dev)
        sqrt_dim = math.sqrt(sd.size(1))
        alpha = their_alpha(args.outputs.format(seed=s, dataset=args.dataset))
        cos_t, cos_c = (cq[qi] * cd[ti]).sum(1), (cq[qi] * cd[ci]).sum(1)
        S_cos = cq[qi] @ cd[test_cols].T
        if s == args.seeds[0]:
            res["cosine (SparseCL's pipeline)"] = dict(pairwise=float((cos_t > cos_c).float().mean()),
                                                      test_only_ndcg10=ndcg10(S_cos, tgt_test))
        # the query-free sparse embedding: the average over the split's queries, as in test_qf.py
        for tag, sq in (("", sq_all), ("_queryfree", sq_all.mean(0, keepdim=True).expand_as(sq_all))):
            ht = hoyer(sd[ti] - sq[qi], sqrt_dim)
            hc = hoyer(sd[ci] - sq[qi], sqrt_dim)
            pair = float(((cos_t + alpha * ht) > (cos_c + alpha * hc)).float().mean())
            S = torch.empty_like(S_cos)
            D = sd[test_cols]
            for a in range(0, len(test_qids), args.chunk):
                b = min(a + args.chunk, len(test_qids))
                S[a:b] = S_cos[a:b] + alpha * hoyer(D[None] - sq[qi[a:b]][:, None], sqrt_dim)
            key = f"sparsecl_s{s}{tag}"
            res[key] = dict(alpha=alpha, pairwise=pair, test_only_ndcg10=ndcg10(S, tgt_test))
            print(key, res[key], flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
