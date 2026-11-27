"""
cr_analysis.py - what does a trained contradiction model's clash encode?

1. Contradiction-style probe (no query involved). Passages that are relevant contradictions in the
   qrels are compared with the queries' own passages (paraphrases of the original):
     best slot AUC   for each slot, its value (one number) or main direction of variation (vector
                     values) as a score; the best slot's AUC at telling contradictions from
                     paraphrases, without seeing any query
     linear probe    a logistic regression on all stored clash numbers of a passage, 2-fold
                     cross-validated AUC: how much of "is this a generated contradiction?" the
                     stored vector carries on its own
2. Single-fact sensitivity. probe_build.py's rules change exactly ONE fact (a year, number, month,
   negation, antonym or name) in a test passage; the control deletes one article instead.
     ratio           mean clash of the single-fact pairs / mean clash of the control pairs
     wins            share of passages where the changed fact clashes more than the control

    python cr_analysis.py --dataset hotpotqa --model runs/cr_hotpotqa_s0/last \\
        --out results/analysis_cr_hotpotqa_s0.json
"""
import argparse
import json
import os
import random

import torch

import probe_build as pb
from cr_data import load_split
from cr_model import SlotEncoder


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_style", type=int, default=4000, help="passages for the style probe")
    ap.add_argument("--n_edit", type=int, default=600, help="passages for the sensitivity test")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def auc(pos, neg):
    pos, neg = pos.flatten(), neg.flatten()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    ranks = torch.empty_like(allv)
    ranks[allv.argsort()] = torch.arange(1, len(allv) + 1, dtype=allv.dtype)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def slots(model, texts, bs, device):
    A, V = [], []
    for i in range(0, len(texts), bs):
        a, v = model(model.tokenize(texts[i:i + bs], device))
        A.append(a.float().cpu())
        V.append(v.float().cpu())
    return torch.cat(A), torch.cat(V)


def linear_probe_auc(X, y, seed):
    """2-fold cross-validated logistic regression AUC on standardised features."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    folds = [perm[: len(y) // 2], perm[len(y) // 2:]]
    scores = torch.zeros(len(y))
    for k in range(2):
        tr, te = folds[1 - k], folds[k]
        mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w = torch.zeros(X.size(1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)
        yt = y[tr].float()

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, yt) + \
                1e-3 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            scores[te] = Xte @ w + b
    return auc(scores[y], scores[~y])


def main():
    args = parse()
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SlotEncoder.load(args.model).to(device).eval()
    corpus, queries, qrels = load_split(args.data_dir, args.dataset, args.split)

    # 1. style probe: relevant contradictions vs the queries' own passages (paraphrases)
    contra = sorted({d for rel in qrels.values() for d, r in rel.items() if r > 0 and d in corpus})
    para = sorted({q for q in qrels if q in corpus and q not in set(contra)})
    if not para:                                    # queries not stored in the corpus: use their text
        para_texts = sorted({queries[q] for q in qrels if q in queries})
    else:
        para_texts = [corpus[q] for q in para]
    contra_texts = [corpus[d] for d in contra]
    rng.shuffle(contra_texts)
    rng.shuffle(para_texts)
    m = min(args.n_style // 2, len(contra_texts), len(para_texts))
    texts = contra_texts[:m] + para_texts[:m]
    y = torch.tensor([True] * m + [False] * m)
    A, V = slots(model, texts, args.bs, device)
    k = V.size(-1)
    per_slot = []
    for c in range(model.n):
        if k == 1:
            s = V[:, c, 0]
        else:
            X = V[:, c] - V[:, c].mean(0, keepdim=True)
            s = X @ torch.linalg.svd(X, full_matrices=False)[2][0]
        a_ = auc(s[y], s[~y])
        per_slot.append(max(a_, 1 - a_))
    stored = model.stored_vec(A, V)
    probe = linear_probe_auc(stored, y, args.seed)
    probe = max(probe, 1 - probe)

    # 2. single-fact sensitivity
    src = [t for t in sorted({queries[q] for q in qrels if q in queries})
           if 15 <= len(t.split()) <= 150]
    rng.shuffle(src)
    pool = {}
    for t in src[:20000]:
        w = t.split()
        for i, j in pb.name_spans(w):
            pool.setdefault(j - i, set()).add(" ".join(pb.split_punct(x)[1] for x in w[i:j]))
    pool = {kk: sorted(v) for kk, v in pool.items()}
    orig, edit, ctrl = [], [], []
    for t in src:
        words = t.split()
        arts = [i for i, w in enumerate(words) if w.lower() in ("the", "a", "an")]
        if not arts:
            continue
        order = list(pb.EDITS)
        rng.shuffle(order)
        for et in order:
            fn = pb.EDITS[et]
            e = fn(words, rng, pool) if et == "entity" else fn(words, rng)
            if e:
                new = pb.apply(words, e)[0]
                if new != words:
                    j = rng.choice(arts)
                    orig.append(t)
                    edit.append(" ".join(new))
                    ctrl.append(" ".join(w for i, w in enumerate(words) if i != j))
                    break
        if len(orig) >= args.n_edit:
            break
    if orig:
        Ao, Vo = slots(model, orig, args.bs, device)
        Ae, Ve = slots(model, edit, args.bs, device)
        Ac, Vc = slots(model, ctrl, args.bs, device)
        k_edit = (Ao * Ae * (Ve - Vo).pow(2).sum(-1)).sum(1)
        k_ctrl = (Ao * Ac * (Vc - Vo).pow(2).sum(-1)).sum(1)
    else:                                           # no passage allowed a rule-based edit
        k_edit = k_ctrl = torch.full((1,), float("nan"))
    res = dict(
        model=args.model, dataset=args.dataset, n_slots=model.n, value_dim=k,
        style_n=2 * m, style_best_slot_auc=max(per_slot), style_slot_aucs=per_slot,
        style_linear_probe_auc=probe,
        sensitivity_n=len(orig),
        sensitivity_ratio=float(k_edit.mean() / k_ctrl.mean().clamp_min(1e-12)),
        sensitivity_wins=float((k_edit > k_ctrl).float().mean()) if len(orig) else float("nan"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"style probe: best slot AUC {res['style_best_slot_auc']:.3f}, linear probe AUC "
          f"{res['style_linear_probe_auc']:.3f} ({2 * m} passages)")
    print(f"single-fact sensitivity: clash ratio {res['sensitivity_ratio']:.1f}x, edit > control in "
          f"{100 * res['sensitivity_wins']:.0f}% of {len(orig)} passages")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
