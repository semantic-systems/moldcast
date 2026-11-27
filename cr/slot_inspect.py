"""
slot_inspect.py - a readable report for reading what each slot does, by hand.

For a sample of corpus passages, every slot's commitment, value and attention are computed. Per slot:
  * value extremes  the passages where the slot takes its HIGHEST and LOWEST values (only where it
                    commits, a > 0.5). One-number slots use the value itself; vector values use
                    their main direction of variation (first principal component over passages).
                    Reading what separates the two lists is the main tool: a slot may encode a
                    topic, a sentiment, a writing style, a kind of fact...
  * commitment      the passages where the slot commits most and least (what makes it speak up)
  * edits carried   from the probe's single-fact edits (if --probe is given), the changes for which
                    this slot carries the largest share of the clash
Snippets show a window around the slot's most-attended words, which are in **bold**.

    python slot_inspect.py --dataset hotpotqa --model runs/v5_hotpotqa_k1_n16_s0/last \\
        --probe data/probe_hotpotqa.jsonl
    -> runs/v5_hotpotqa_k1_n16_s0/last/slot_inspect_hotpotqa.md
"""
import argparse
import json
import os
import random
from collections import defaultdict

import torch

from cr_data import load_split
from cr_model import SlotEncoder


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default=None, help="probe_build.py output, for 'edits carried'")
    ap.add_argument("--n_passages", type=int, default=2000)
    ap.add_argument("--max_words", type=int, default=150)
    ap.add_argument("--n_show", type=int, default=8, help="passages per list")
    ap.add_argument("--window", type=int, default=12, help="words of context around bold words")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


@torch.no_grad()
def encode(model, words, device):
    """-> a (n,), v (n, k), per-slot word weights (n, n_words)."""
    enc = model.tok(words, is_split_into_words=True, truncation=True,
                    max_length=model.config["max_len"], return_tensors="pt")
    b = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    a, v, w = model(b, return_attn=True)
    ww = torch.zeros(model.n, len(words))
    for t, wid in enumerate(enc.word_ids()):
        if wid is not None:
            ww[:, wid] += w[0, :, t].float().cpu()
    return a[0].float().cpu(), v[0].float().cpu(), ww


def snippet(words, weights, window, n_bold=3):
    """a window of words around the most-weighted ones, those in bold."""
    if weights.sum() <= 0:
        return " ".join(words[:2 * window]) + (" ..." if len(words) > 2 * window else "")
    top = set(weights.topk(min(n_bold, len(words))).indices.tolist())
    lo = max(0, min(top) - window // 2)
    hi = min(len(words), max(max(top) + window // 2 + 1, lo + 2 * window))
    out = [f"**{w}**" if i in top else w for i, w in enumerate(words[lo:hi], start=lo)]
    return ("... " if lo > 0 else "") + " ".join(out) + (" ..." if hi < len(words) else "")


def main():
    args = parse()
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SlotEncoder.load(args.model).to(device).eval()
    corpus, _, _ = load_split(args.data_dir, args.dataset, args.split)
    texts = sorted({t for t in corpus.values() if 5 <= len(t.split()) <= args.max_words})
    rng.shuffle(texts)
    texts = texts[:args.n_passages]
    recs = []
    for t in texts:
        words = t.split()
        a, v, w = encode(model, words, device)
        recs.append((words, a, v, w))
    A = torch.stack([r[1] for r in recs])                       # (N, n)
    V = torch.stack([r[2] for r in recs])                       # (N, n, k)
    n, k = model.n, V.size(-1)

    # one score per passage and slot: the value, or its first principal component
    score = torch.zeros(len(recs), n)
    explained = {}
    for c in range(n):
        if k == 1:
            score[:, c] = V[:, c, 0]
            explained[c] = 1.0
        else:
            X = V[:, c] - V[:, c].mean(0, keepdim=True)
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            score[:, c] = X @ Vt[0]
            explained[c] = float(S[0] ** 2 / (S ** 2).sum().clamp_min(1e-12))

    carried = defaultdict(list)
    if args.probe:
        for line in open(args.probe):
            r = json.loads(line)
            o, e = r["original"].split(), r["edited"].split()
            ao, vo, _ = encode(model, o, device)
            ae, ve, _ = encode(model, e, device)
            per = ao * ae * (ve - vo).pow(2).sum(-1)
            if per.sum() <= 0:
                continue
            s = int(per.argmax())
            carried[s].append((float(per[s] / per.sum()), r["type"],
                               " ".join(o[i] for i in r["span_original"]) + " -> " +
                               " ".join(e[i] for i in r["span_edited"])))

    lines = [f"# Slot inspection: {args.model} ({args.dataset} {args.split})", "",
             f"{len(recs)} passages, {n} slots, {k} number(s) per value. Bold words: the slot's "
             f"most-attended words in each passage.", ""]
    order = sorted(range(n), key=lambda c: -float(A[:, c].mean()))
    for c in order:
        com = A[:, c]
        active = (com > 0.5).nonzero().flatten().tolist()
        lines += [f"## Slot {c}", "",
                  f"commits (a > 0.5) in {100 * len(active) / len(recs):.0f}% of passages; "
                  f"mean commitment {float(com.mean()):.2f}"
                  + (f"; first component explains {100 * explained[c]:.0f}% of its value "
                     f"variation" if k > 1 else ""), ""]
        if len(active) >= 2 * args.n_show:
            srt = sorted(active, key=lambda i: float(score[i, c]))
            for title, idx in (("Highest values", srt[::-1][:args.n_show]),
                               ("Lowest values", srt[:args.n_show])):
                lines += [f"**{title}**", ""]
                for i in idx:
                    words, _, _, w = recs[i]
                    lines.append(f"- ({float(score[i, c]):+.2f}) " +
                                 snippet(words, w[c], args.window))
                lines.append("")
        else:
            lines += ["(too few committed passages for value extremes)", ""]
        srt_a = sorted(range(len(recs)), key=lambda i: float(com[i]))
        lines += ["**Most committed**", ""]
        for i in srt_a[::-1][:max(3, args.n_show // 2)]:
            words, _, _, w = recs[i]
            lines.append(f"- (a = {float(com[i]):.2f}) " + snippet(words, w[c], args.window))
        lines += ["", "**Least committed**", ""]
        for i in srt_a[:max(3, args.n_show // 2)]:
            words, _, _, w = recs[i]
            lines.append(f"- (a = {float(com[i]):.2f}) " + snippet(words, w[c], args.window))
        lines.append("")
        if args.probe:
            ex = sorted(carried.get(c, []), reverse=True)
            kinds = defaultdict(int)
            for _, t, _ in ex:
                kinds[t] += 1
            lines += [f"**Single-fact edits carried** ({len(ex)}; by kind: " +
                      ", ".join(f"{t} {m}" for t, m in sorted(kinds.items(), key=lambda x: -x[1]))
                      + ")", ""]
            for share, t, ch in ex[:args.n_show]:
                lines.append(f"- [{t}] {ch} ({100 * share:.0f}% of the clash)")
            lines.append("")
    text = "\n".join(lines)
    out = args.out or os.path.join(args.model, f"slot_inspect_{args.dataset}.md")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(text[:3000] + ("\n..." if len(text) > 3000 else ""))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
