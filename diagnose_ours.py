"""
diagnose_ours.py - two diagnostics of our trained clash models on the VitaminC test split
(evaluation only; vectors built exactly as cr/cr_eval.py builds them):

  pairwise    share of test queries whose target scores above its confounder (the sentence the
              query paraphrases). A score that ignores the query can reach at most 50% here.
  test-only   NDCG@10 against the test split's own passages only (no train/dev passages), by exact
              scoring; a gain that comes from recognising passages seen in training disappears here.

Both normally and query-free (every query's clash vector replaced by the average one), with the
alpha chosen on dev by cr_eval.py (the exact-search protocol: alpha_vitaminc_full*.json). Zero-shot
cosine is reported alongside.

    python diagnose_ours.py --data data --runs cr/runs --out results/diagnose_ours.json
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cr"))
from cr_data import load_split          # noqa: E402
from cr_eval import cached_cos          # noqa: E402
from cr_model import SlotEncoder        # noqa: E402

SEEDS = (0, 1, 2)


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="cr/runs")
    ap.add_argument("--cos_model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--arms", nargs="+", default=["cr16", "cr2"], help="configurations (run names)")
    ap.add_argument("--dataset", default="vitaminc", help="the benchmark (vitaminc)")
    ap.add_argument("--out", default="results/diagnose_ours.json")
    # cached_cos reads these from its args
    ap.add_argument("--cache_dir", default="cr/cache")
    return ap.parse_args()


def triples(qids, qrels, pos):
    """(query index, target index, confounder index) for every test query."""
    out = []
    for i, q in enumerate(qids):
        (t,) = qrels[q]
        c = q.replace("-q", "-", 1)               # test-q7-a -> test-7-a: the sentence it paraphrases
        out.append((i, pos[t], pos[c]))
    return out


def ndcg10(S, target_cols):
    """one relevant passage per query: NDCG@10 = 1 / log2(rank + 1) if ranked in the top 10."""
    top = S.topk(10, dim=1).indices
    hit = top == target_cols[:, None]
    rank = hit.float().argmax(1) + 1
    return float(torch.where(hit.any(1), 1.0 / torch.log2(rank.float() + 1), torch.zeros_like(rank,
                 dtype=torch.float)).mean())


def measures(score_pairs, S_test, tgt_cols):
    t, c = score_pairs
    return dict(pairwise=float((t > c).float().mean()), test_only_ndcg10=ndcg10(S_test, tgt_cols))


def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    corpus, queries, qrels = load_split(args.data, args.dataset, "test")
    qids = [q for q in qrels if q in queries]
    pids = list(corpus)
    pos = {p: i for i, p in enumerate(pids)}
    tri = triples(qids, qrels, pos)
    test_cols = [i for i, p in enumerate(pids) if p.startswith("test-")]
    col_in_test = {c: j for j, c in enumerate(test_cols)}
    qi = torch.tensor([a for a, _, _ in tri], device=dev)
    ti = torch.tensor([b for _, b, _ in tri], device=dev)
    ci = torch.tensor([c for _, _, c in tri], device=dev)
    tgt_test = torch.tensor([col_in_test[b] for _, b, _ in tri]).to(dev)
    ptext, qtext = [corpus[p] for p in pids], [queries[q] for q in qids]
    print(f"{args.dataset} test: {len(qids):,} queries, corpus {len(pids):,} ({len(test_cols):,} test passages)")

    Cc = cached_cos(args.cos_model, ptext, args.max_len, args, dev, amp).to(dev).float()
    Qc = cached_cos(args.cos_model, qtext, args.max_len, args, dev, amp).to(dev).float()
    cos_pair = ((Qc[qi] * Cc[ti]).sum(1), (Qc[qi] * Cc[ci]).sum(1))
    S_cos = Qc @ Cc[test_cols].T
    res = {"cosine": measures(cos_pair, S_cos, tgt_test)}
    print("cosine", res["cosine"], flush=True)

    for arm in args.arms:
        for s in SEEDS:
            path = os.path.join(args.runs, f"{arm}_{args.dataset}_s{s}", "last")
            if not os.path.exists(os.path.join(path, "model.pt")):
                continue
            model = SlotEncoder.load(path).to(dev).eval()
            with torch.no_grad():
                # the encoder returns its vectors on the CPU: bring them next to the cosine ones
                Ck = model.gate_vectors(model.encode(ptext, bs=args.bs, device=dev, amp=amp,
                                                     side="stored").to(dev), Cc).float()
                Qk = (model.gate_vectors(model.encode(qtext, bs=args.bs, device=dev, amp=amp,
                                                      side="query").to(dev), Qc)
                      * model.k_scale()).float()
            if Qk.dim() != 2:
                raise SystemExit("diagnose_ours.py expects ungated clash vectors (--gate_dim 0)")
            for tag, Q in (("", Qk), ("_queryfree", Qk.mean(0, keepdim=True).expand_as(Qk))):
                a_file = os.path.join(path, f"alpha_{args.dataset}_full{tag}.json")
                alpha = json.load(open(a_file))["alpha"]
                pair = (cos_pair[0] + alpha * (Q[qi] * Ck[ti]).sum(1),
                        cos_pair[1] + alpha * (Q[qi] * Ck[ci]).sum(1))
                S = S_cos + alpha * (Q @ Ck[test_cols].T)
                key = f"{arm}_s{s}{tag}"
                res[key] = dict(alpha=alpha, **measures(pair, S, tgt_test))
                print(key, res[key], flush=True)
            del model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
