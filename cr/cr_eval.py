"""
cr_eval.py - contradiction retrieval on the SparseCL benchmarks, with their protocol.

Protocol (as in SparseCL's test_contradiction_faiss_final.py):
  * corpus = union of the train, dev and test corpora; queries and qrels from --split;
  * self-matches are handled as in SparseCL (--exclusion sparsecl, the default):
      - cosine row: no exclusion before the top 10 (their FAISS search returns everything);
      - rows that use the trained model: passages with text identical to the query are excluded
        before the top 10 (SparseCL's Hoyer term is -1e9 when the two embeddings coincide);
      - then, as BEIR's evaluate does, a result whose passage id equals the query id is dropped
        from the top 10 (that query is scored on 9 results).
    --exclusion strict excludes identical text and identical id before the top 10 in every row;
  * score F(q, d) = cos(q, d) + alpha * K~(q, d); top 10 per query; NDCG@10 and Recall@10;
  * alpha is chosen on the dev split by their interval search over [0, 10] (10 bins, refined until
    the interval is narrower than 0.01) and saved; the test split reuses the dev alpha.

Search modes (--protocol):
  full     exact scores against EVERY corpus passage: F is one inner product, so this is a
           single matrix product per query block (what an inner-product index does exactly)
  rerank   SparseCL's procedure: cosine top 1,000 (exact here; they use HNSW), then rerank by F

Also reported: cosine alone (alpha = 0) and K alone, on the same candidates.

--query_free: every query's clash vector is replaced by the AVERAGE query clash vector, so the clash
becomes a property of the passage alone (cosine still compares query and passage). If this keeps
most of the score, the benchmark can largely be solved by recognising what contradiction passages
look like, without comparing them with the query. Alpha is tuned on dev in the same mode.

    python cr_eval.py --dataset hotpotqa --split dev  --model runs/hotpotqa_s0/last
    python cr_eval.py --dataset hotpotqa --split test --model runs/hotpotqa_s0/last
    python cr_eval.py --dataset hotpotqa --split test                 # cosine only, no model
"""
import argparse
import hashlib
import json
import os

import torch

from cr_data import DATASETS, evaluate, load_split
from cr_model import CosEncoder, SlotEncoder


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", default=None, help="a run's last/ or best/ directory")
    ap.add_argument("--cos_model", default=None,
                    help="frozen cosine encoder (default: the one the model was trained with)")
    ap.add_argument("--protocol", choices=["full", "rerank", "both"], default="both")
    ap.add_argument("--k_rerank", type=int, default=1000)
    ap.add_argument("--exclusion", choices=["sparsecl", "strict"], default="sparsecl")
    ap.add_argument("--query_free", action="store_true",
                    help="replace each query's clash vector by the average query clash vector")
    ap.add_argument("--alpha", type=float, default=None, help="skip the search, use this alpha")
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def cached_cos(cos_name, texts, max_len, args, device, amp):
    key = hashlib.md5(("\n".join(texts) + cos_name + str(max_len)).encode()).hexdigest()[:16]
    path = os.path.join(args.cache_dir, f"cos_{os.path.basename(cos_name)}_{key}.pt")
    if os.path.exists(path):
        return torch.load(path)
    enc = CosEncoder(cos_name, max_len=max_len)
    e = enc.encode(texts, bs=args.bs, device=device, amp=amp)
    os.makedirs(args.cache_dir, exist_ok=True)
    torch.save(e, path)
    return e


class Scorer:
    """Holds query / corpus vectors and returns top-10 results for any alpha."""

    def __init__(self, Qc, Cc, Qk, Ck, same_text, same_id, exclusion, k_rerank, device):
        self.dev = device
        self.Qc, self.Cc = Qc.to(device), Cc.to(device)
        self.Qk = Qk.to(device) if Qk is not None else None
        self.Ck = Ck.to(device) if Ck is not None else None
        strict = exclusion == "strict"
        both = [sorted(set(t) | set(i)) for t, i in zip(same_text, same_id)]
        # corpus indices excluded before the top 10, per query, for each kind of row
        self.excl = {"cos": both if strict else [[] for _ in same_text],
                     "model": both if strict else same_text}
        self.k_rerank = k_rerank
        self._cand = None

    def _mask(self, S, rows, kind):
        for r, q in enumerate(rows):
            if self.excl[kind][q]:
                S[r, self.excl[kind][q]] = -1e9
        return S

    def cand(self):
        """cosine shortlist of k_rerank per query, computed once (as SparseCL's FAISS search:
        nothing excluded, unless --exclusion strict)."""
        if self._cand is None:
            out, nq = [], self.Qc.size(0)
            for s in range(0, nq, 256):
                S = self._mask(self.Qc[s:s + 256] @ self.Cc.T, range(s, min(s + 256, nq)), "cos")
                out.append(S.topk(min(self.k_rerank, S.size(1)), dim=1).indices)
            self._cand = torch.cat(out)
        return self._cand

    def top10(self, alpha, protocol, k_only=False):
        """-> (Q, 10) corpus indices and scores. alpha = 0 without k_only is the cosine row."""
        kind = "model" if (k_only or (alpha and self.Qk is not None)) else "cos"
        idx_all, sc_all = [], []
        nq = self.Qc.size(0)
        for s in range(0, nq, 256):
            rows = range(s, min(s + 256, nq))
            if protocol == "full":
                S = torch.zeros(len(rows), self.Cc.size(0), device=self.dev) if k_only \
                    else self.Qc[s:s + 256] @ self.Cc.T
                if kind == "model":
                    S = S + (1.0 if k_only else alpha) * (self.Qk[s:s + 256] @ self.Ck.T)
                S = self._mask(S, rows, kind)
                v, i = S.topk(10, dim=1)
            else:
                c = self.cand()[s:s + 256]                                  # (b, k)
                S = torch.zeros(c.shape, device=self.dev) if k_only \
                    else torch.einsum("bd,bkd->bk", self.Qc[s:s + 256], self.Cc[c])
                if kind == "model":
                    S = S + (1.0 if k_only else alpha) * torch.einsum(
                        "bd,bkd->bk", self.Qk[s:s + 256], self.Ck[c])
                for r, q in enumerate(rows):            # exclusions also apply after rescoring
                    if self.excl[kind][q]:
                        bad = torch.isin(c[r], torch.tensor(self.excl[kind][q], device=c.device))
                        S[r, bad] = -1e9
                v, j = S.topk(10, dim=1)
                i = c.gather(1, j)
            idx_all.append(i.cpu())
            sc_all.append(v.cpu())
        return torch.cat(idx_all), torch.cat(sc_all)


def to_results(qids, pids, idx, sc):
    return {q: {pids[j]: float(s) for j, s in zip(idx[r].tolist(), sc[r].tolist())}
            for r, q in enumerate(qids)}


def interval_search(fn, lo=0.0, hi=10.0, bins=10, eps=0.01):
    """SparseCL's hyper_parameter_selection: evaluate bin midpoints, zoom into the best bin."""
    best_overall = (-1.0, None)
    while hi - lo > eps:
        best = (-1.0, (lo, lo))
        for b in range(bins):
            a = lo + (b + 0.5) * (hi - lo) / bins
            v = fn(a)
            if v > best[0]:
                best = (v, (lo + b * (hi - lo) / bins, lo + (b + 1) * (hi - lo) / bins))
            if v > best_overall[0]:
                best_overall = (v, a)
        lo, hi = best[1]
    return lo, best_overall


def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    corpus, queries, qrels = load_split(args.data_dir, args.dataset, args.split)
    qids = [q for q in qrels if q in queries]
    pids = list(corpus)
    ptext = [corpus[p] for p in pids]
    qtext = [queries[q] for q in qids]
    print(f"{args.dataset} {args.split}: {len(qids):,} queries, corpus {len(pids):,}")
    pos_of_id = {p: i for i, p in enumerate(pids)}
    by_text = {}
    for i, t in enumerate(ptext):
        by_text.setdefault(t, []).append(i)
    same_text = [by_text.get(t, []) for t in qtext]
    same_id = [[pos_of_id[q]] if q in pos_of_id else [] for q in qids]
    n_self = sum(1 for t in same_text if t)
    print(f"queries whose text is in the corpus: {n_self:,}; whose id is a corpus id: "
          f"{sum(1 for i in same_id if i):,} (exclusion: {args.exclusion})")

    model, cfg = None, {}
    if args.model:
        model = SlotEncoder.load(args.model)
        targs = os.path.join(os.path.dirname(args.model.rstrip("/")), "train_args.json")
        cfg = json.load(open(targs)) if os.path.exists(targs) else {}
    cos_name = args.cos_model or cfg.get("cos_model", "BAAI/bge-base-en-v1.5")
    max_len = args.max_len or cfg.get("max_len") or (512 if args.dataset == "arguana" else 256)

    Cc = cached_cos(cos_name, ptext, max_len, args, device, amp)
    Qc = cached_cos(cos_name, qtext, max_len, args, device, amp)
    Qk = Ck = None
    if model is not None:
        with torch.no_grad():
            Ck = model.gate_vectors(
                model.encode(ptext, bs=args.bs, device=device, amp=amp, side="stored"), Cc)
            Qk = model.gate_vectors(
                model.encode(qtext, bs=args.bs, device=device, amp=amp, side="query"), Qc) \
                * model.k_scale()
        print(f"clash vectors: {Ck.size(1):,} numbers per passage")
        if args.query_free:
            Qk = Qk.mean(0, keepdim=True).expand_as(Qk).contiguous()
            print("query-free: every query uses the average query clash vector")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    sc = Scorer(Qc, Cc, Qk, Ck, same_text, same_id, args.exclusion, args.k_rerank, device)

    protocols = ["full", "rerank"] if args.protocol == "both" else [args.protocol]
    report = dict(dataset=args.dataset, split=args.split, model=args.model, cos_model=cos_name,
                  n_queries=len(qids), corpus=len(pids), exclusion=args.exclusion,
                  query_free=args.query_free, results={})
    for proto in protocols:
        r = report["results"].setdefault(proto, {})
        r["cosine"] = evaluate(qrels, to_results(qids, pids, *sc.top10(0.0, proto)))
        if Qk is None:
            continue
        r["K_only"] = evaluate(qrels, to_results(qids, pids, *sc.top10(0.0, proto, k_only=True)))
        tag = ("" if args.exclusion == "sparsecl" else f"_{args.exclusion}") + \
            ("_queryfree" if args.query_free else "")
        alpha_file = os.path.join(args.model, f"alpha_{args.dataset}_{proto}{tag}.json")
        alpha = args.alpha
        if alpha is None and args.split == "test" and os.path.exists(alpha_file):
            alpha = json.load(open(alpha_file))["alpha"]
            print(f"{proto}: alpha {alpha:.4f} from {alpha_file}")
        if alpha is None:
            fn = lambda a: evaluate(qrels, to_results(qids, pids, *sc.top10(a, proto)))["NDCG@10"]
            alpha, (best_v, best_a) = interval_search(fn)
            print(f"{proto}: alpha {alpha:.4f} chosen on {args.split} (best probe {best_v:.4f} "
                  f"at {best_a:.3f})")
            if args.split == "dev":
                with open(alpha_file, "w") as f:
                    json.dump({"alpha": alpha, "dev_ndcg10_probe": best_v}, f)
            else:
                print("  warning: no dev alpha found, alpha was tuned on the test split itself")
        r["alpha"] = alpha
        r["cos_plus_K"] = evaluate(qrels, to_results(qids, pids, *sc.top10(alpha, proto)))
    for proto, r in report["results"].items():
        line = "  ".join(f"{k} {v['NDCG@10']:.4f}" for k, v in r.items() if isinstance(v, dict))
        print(f"{args.dataset} {args.split} [{proto}] NDCG@10: {line}"
              + (f"  (alpha {r['alpha']:.3f})" if "alpha" in r else ""))
    tag = ("" if args.exclusion == "sparsecl" else f"_{args.exclusion}") + \
        ("_queryfree" if args.query_free else "")
    out = args.out or (os.path.join(args.model, f"eval_{args.dataset}_{args.split}{tag}.json")
                       if args.model else f"results/cos_{args.dataset}_{args.split}{tag}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
