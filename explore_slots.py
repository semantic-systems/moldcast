"""
explore_slots.py - an open-ended look at what each slot of a trained MOLDCAST model responds to,
on real text (every passage of the VitaminC corpus: Wikipedia sentences). Exploratory: no criteria
are fixed in advance, so the report shows ALL slots of every model, and themes should recur across
seeds (slots are not aligned between seeds) before they are trusted.

For each slot:
  triggers   the sentences that make it commit (a > 0.5) against all sentences: the most
             distinctive words (log-odds with an informative Dirichlet prior, Monroe et al. 2008;
             z-scores, words seen at least --min_count times) and how much more often they contain
             years, numbers, money, percentages, dates, names or negations
  values     k-means clusters of the slot's values over the sentences it commits to, with each
             cluster's distinctive words (against the slot's other committed sentences) and its
             most central sentences
  examples   the sentences it commits to most strongly

    python explore_slots.py --arm cr16             # writes results/explore_cr16.md and .json
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cr"))
from cr_data import load_split          # noqa: E402
from cr_model import SlotEncoder        # noqa: E402

TOKEN = re.compile(r"[a-z][a-z'\-]+|\d+(?:[.,]\d+)*")
STOP = set("""a an the of in on at to for from by with and or but as is was were be been are it its
this that these those he she they his her their which who whom whose what when where how than then
also has have had not no into over after before about between during under while s""".split())
MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
FEATURES = {
    "year": re.compile(r"\b(1[5-9]\d\d|20\d\d)\b"),
    "number": re.compile(r"\d"),
    "money": re.compile(r"\$|\bmillion\b|\bbillion\b|\bgross", re.I),
    "percent": re.compile(r"%|\bpercent\b", re.I),
    "date": re.compile(MONTHS),
    "name": re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+"),
    "negation": re.compile(r"\b(not|no|never|none|nor|without|n't|cannot)\b", re.I),
}


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="cr/runs")
    ap.add_argument("--arm", default="cr16")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--clusters", type=int, default=4)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--top_words", type=int, default=10)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def words(text):
    return [w for w in TOKEN.findall(text.lower()) if w not in STOP]


def distinctive(sub_counts, ref_counts, min_count, top, prior_scale=0.01):
    """log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008): words over-
    represented in sub against ref, as z-scores; only words seen at least min_count times in sub."""
    n_sub, n_ref = sum(sub_counts.values()), sum(ref_counts.values())
    prior = {w: c * prior_scale for w, c in ref_counts.items()}
    a0 = sum(prior.values())
    out = []
    for w, cs in sub_counts.items():
        if cs < min_count:
            continue
        aw, cr = prior.get(w, prior_scale), ref_counts.get(w, 0)
        l1 = math.log((cs + aw) / (n_sub + a0 - cs - aw))
        l2 = math.log((cr + aw) / (n_ref + a0 - cr - aw))
        var = 1 / (cs + aw) + 1 / (cr + aw)
        out.append((w, (l1 - l2) / math.sqrt(var)))
    return sorted(out, key=lambda x: -x[1])[:top]


def kmeans(X, k, iters=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab, C


@torch.no_grad()
def slots(model, texts, bs, dev):
    A, V = [], []
    for s in range(0, len(texts), bs):
        b = model.tokenize(texts[s:s + bs], dev)
        with torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16, enabled=dev == "cuda"):
            a, v = model.run_head(model.hidden(b), b)
        A.append(a.float().cpu())
        V.append(v.float().cpu())
    return torch.cat(A), torch.cat(V)


def analyse(seed, args, texts, toks, feats, all_counts, dev):
    model = SlotEncoder.load(os.path.join(args.runs, f"{args.arm}_vitaminc_s{seed}", "last")).to(dev).eval()
    a, v = slots(model, texts, args.bs, dev)
    n = a.size(1)
    base = {f: sum(x[f] for x in feats) / len(feats) for f in FEATURES}
    res = []
    for c in range(n):
        on = (a[:, c] > 0.5).nonzero().flatten().tolist()
        r = dict(slot=c, commit=len(on) / len(texts))
        if 0 < len(on) < len(texts):
            sub = Counter(w for i in on for w in toks[i])
            r["words"] = distinctive(sub, all_counts, args.min_count, args.top_words)
            r["features"] = {f: (sum(feats[i][f] for i in on) / len(on)) / max(base[f], 1e-9)
                             for f in FEATURES}
        top = a[:, c].argsort(descending=True)[:args.examples].tolist()
        r["examples"] = [texts[i] for i in top]
        if len(on) >= 50 * args.clusters:
            X = v[on, c]
            X = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)
            lab, C = kmeans(X, args.clusters, seed=seed)
            sub_all = Counter(w for i in on for w in toks[i])
            cl = []
            for j in range(args.clusters):
                members = [on[i] for i in (lab == j).nonzero().flatten().tolist()]
                if not members:
                    continue
                dist = torch.cdist(X[lab == j], C[j:j + 1]).flatten()
                central = [members[i] for i in dist.argsort()[:args.examples].tolist()]
                cnt = Counter(w for i in members for w in toks[i])
                cl.append(dict(size=len(members) / len(on),
                               words=distinctive(cnt, sub_all, max(5, args.min_count // 4), 6),
                               examples=[texts[i] for i in central]))
            r["clusters"] = cl
        res.append(r)
    return dict(seed=seed, slots=res, commit_rate=(a > 0.5).float().mean().item())


def report(arm, runs, n_texts):
    L = [f"# What the slots of {arm} respond to (exploratory; {n_texts:,} VitaminC sentences)", "",
         "All slots are shown. Words: the most distinctive words of the sentences a slot commits to "
         "(z-score). Features: how much more (or less) often those sentences contain each feature "
         "than all sentences (1.0 = as often). Themes should recur across seeds before they are trusted."]
    for r in runs:
        L += ["", f"## Seed {r['seed']} (overall commitment rate {100 * r['commit_rate']:.0f}%)"]
        for s in r["slots"]:
            L += ["", f"### Slot {s['slot']}: commits on {100 * s['commit']:.0f}% of sentences"]
            if "words" in s:
                L.append("Words: " + (", ".join(f"{w} ({z:.1f})" for w, z in s["words"]) or "(none distinctive)"))
                L.append("Features: " + ", ".join(f"{f} x{x:.1f}" for f, x in s["features"].items()))
            for e in s["examples"]:
                L.append(f"- {e}")
            for j, c in enumerate(s.get("clusters", [])):
                L.append(f"  - value cluster {j + 1} ({100 * c['size']:.0f}%): "
                         + (", ".join(w for w, _ in c["words"]) or "(no distinctive words)")
                         + f". E.g.: {c['examples'][0]}")
    return "\n".join(L) + "\n"


def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus, _, _ = load_split(args.data, "vitaminc", "test")      # the corpus is the union of all splits
    texts = [corpus[p] for p in corpus]
    toks = [words(t) for t in texts]
    feats = [{f: bool(rx.search(t)) for f, rx in FEATURES.items()} for t in texts]
    all_counts = Counter(w for ws in toks for w in ws)
    runs = []
    for s in args.seeds:
        if os.path.exists(os.path.join(args.runs, f"{args.arm}_vitaminc_s{s}", "last", "model.pt")):
            runs.append(analyse(s, args, texts, toks, feats, all_counts, dev))
            print(f"seed {s}: done (commitment rate {100 * runs[-1]['commit_rate']:.0f}%)", flush=True)
    out = args.out or f"results/explore_{args.arm}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(dict(arm=args.arm, runs=runs), open(out, "w"), indent=1)
    open(out.replace(".json", ".md"), "w").write(report(args.arm, runs, len(texts)))
    print(f"saved {out} and {out.replace('.json', '.md')}")


if __name__ == "__main__":
    main()
