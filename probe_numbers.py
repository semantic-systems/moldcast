"""
probe_numbers.py - a controlled test of what the exploration suggested: that MOLDCAST's slot values
encode the numbers a sentence mentions, as fingerprints that unify digits and number words.

Five sentence templates, each with the numbers 1-100 written as digits and as words ("The town had
6 residents in 2010." / "The town had six residents in 2010."): 1,000 sentences. Distances are the
model's own clash, slot by slot, K_c = a_c a'_c ||v_c - v'_c||^2, summed over slots. Per model:

  digit-word     for each number word, is the digit sentence (same template) with the smallest clash
                 the same number? (top-1 accuracy over 100 numbers; chance 1%); and the mean clash
                 between "6" and "six" relative to the mean clash between different numbers
  last digit     mean clash between different numbers that share a last digit (6 / 16), relative to
                 pairs that do not
  magnitude      Spearman correlation between the clash of two numbers and |log a - log b|
  same number    mean clash between the same number in different templates, relative to different
                 numbers in different templates (does the fingerprint survive a change of context?)
  slots          per slot: how much its commitment depends on the number (standard deviation over
                 the 100 numbers) and the numbers for which it commits most

Writes results/probe_numbers_<arm>.json, a readable .md and, if matplotlib is installed, a figure
of the digit-digit clash matrix (mean over templates), results/probe_numbers_<arm>.png.

    python probe_numbers.py --arm cr16
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cr"))
from cr_model import SlotEncoder        # noqa: E402

TEMPLATES = ["The town had {} residents in 2010.",
             "The film grossed {} million dollars in its opening week.",
             "The team won {} games that season.",
             "The album sold {} copies in its first month.",
             "The building has {} floors and a rooftop garden."]
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
NUMS = list(range(1, 101))


def word(n):
    if n < 20:
        return ONES[n]
    if n == 100:
        return "one hundred"
    t, o = divmod(n, 10)
    return TENS[t] + ("" if o == 0 else "-" + ONES[o])


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="cr/runs")
    ap.add_argument("--arm", default="cr16")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


@torch.no_grad()
def slot_outputs(model, texts, bs, dev):
    A, V = [], []
    for s in range(0, len(texts), bs):
        b = model.tokenize(texts[s:s + bs], dev)
        with torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16, enabled=dev == "cuda"):
            a, v = model.run_head(model.hidden(b), b)
        A.append(a.float().cpu())
        V.append(v.float().cpu())
    return torch.cat(A), torch.cat(V)


def clash(a1, v1, a2, v2):
    """(N1, n), (N1, n, k), (N2, n), (N2, n, k) -> (N1, N2, n): per-slot clash of every pair."""
    return a1[:, None, :] * a2[None, :, :] * (v1[:, None] - v2[None]).pow(2).sum(-1)


def spearman(x, y):
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


def analyse(model, dev, bs):
    T, N = len(TEMPLATES), len(NUMS)
    texts = [t.format(n) for t in TEMPLATES for n in NUMS] + [t.format(word(n)) for t in TEMPLATES for n in NUMS]
    a, v = slot_outputs(model, texts, bs, dev)
    n_slots = a.size(1)
    ad, vd = a[:T * N].view(T, N, n_slots), v[:T * N].view(T, N, n_slots, -1)        # digits
    aw, vw = a[T * N:].view(T, N, n_slots), v[T * N:].view(T, N, n_slots, -1)        # words
    off = ~torch.eye(N, dtype=torch.bool)
    nums = torch.tensor(NUMS)
    same_last = (nums[:, None] % 10 == nums[None] % 10) & off
    logdist = (nums[:, None].float().log() - nums[None].float().log()).abs()
    top1, dw_ratio, ld_ratio, mag, D = [], [], [], [], torch.zeros(N, N)
    for t in range(T):
        Kdd = clash(ad[t], vd[t], ad[t], vd[t]).sum(-1)                               # digits x digits
        Kwd = clash(aw[t], vw[t], ad[t], vd[t]).sum(-1)                               # words x digits
        D += Kdd / T
        top1.append(float((Kwd.argmin(1) == torch.arange(N)).float().mean()))
        dw_ratio.append(float(Kwd.diagonal().mean() / Kwd[off].mean().clamp_min(1e-12)))
        ld_ratio.append(float(Kdd[same_last].mean() / Kdd[off & ~same_last].mean().clamp_min(1e-12)))
        mag.append(spearman(Kdd[off], logdist[off]))
    # the same number in different templates against different numbers in different templates
    same, diff = [], []
    for t1 in range(T):
        for t2 in range(T):
            if t1 != t2:
                K = clash(ad[t1], vd[t1], ad[t2], vd[t2]).sum(-1)
                same.append(K.diagonal().mean())
                diff.append(K[off].mean())
    cross_ratio = float(torch.stack(same).mean() / torch.stack(diff).mean().clamp_min(1e-12))
    # slots: how strongly commitment depends on the number (digits, mean over templates)
    commit = ad.mean(0)                                                               # (N, n)
    slots = []
    for c in range(n_slots):
        col = commit[:, c]
        top = col.argsort(descending=True)[:8].tolist()
        slots.append(dict(slot=c, mean_commit=float(col.mean()), sd_over_numbers=float(col.std()),
                          top_numbers=[NUMS[i] for i in top]))
    m = lambda x: sum(x) / len(x)
    return dict(digit_word_top1=m(top1), digit_word_ratio=m(dw_ratio), last_digit_ratio=m(ld_ratio),
                magnitude_spearman=m(mag), same_number_other_template_ratio=cross_ratio,
                slots=slots), D


def report(arm, runs):
    L = [f"# How {arm} encodes numbers: a controlled probe", "",
         f"{len(TEMPLATES)} templates x numbers 1-100 as digits and as words. Clash ratios below 1 mean the "
         "pairs are closer than the reference pairs.", "",
         "| Seed | Word finds its digit (top-1; chance 1%) | Clash six/6 vs different numbers | Clash 6/16 vs other pairs "
         "| Clash vs log distance (Spearman) | Same number, other template vs different number |",
         "| --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        L.append(f"| {r['seed']} | {100 * r['digit_word_top1']:.0f}% | {r['digit_word_ratio']:.2f} | "
                 f"{r['last_digit_ratio']:.2f} | {r['magnitude_spearman']:.2f} | {r['same_number_other_template_ratio']:.2f} |")
    for r in runs:
        L += ["", f"## Seed {r['seed']}: does each slot's commitment depend on the number?", "",
              "| Slot | Mean commitment | SD over numbers | Numbers it commits to most |", "| --- | --- | --- | --- |"]
        for s in sorted(r["slots"], key=lambda s: -s["sd_over_numbers"]):
            L.append(f"| {s['slot']} | {100 * s['mean_commit']:.0f}% | {100 * s['sd_over_numbers']:.1f} | "
                     f"{', '.join(map(str, s['top_numbers']))} |")
    return "\n".join(L) + "\n"


def figure(arm, mats, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: no figure (pip install matplotlib)")
        return
    fig, axes = plt.subplots(1, len(mats), figsize=(4.2 * len(mats), 4), squeeze=False)
    for ax, (seed, D) in zip(axes[0], mats):
        im = ax.imshow(D.numpy(), cmap="viridis", origin="lower", extent=[0.5, 100.5, 0.5, 100.5])
        ax.set_title(f"{arm}, seed {seed}", fontsize=9)
        ax.set_xlabel("number")
        ax.set_ylabel("number")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Clash between sentences that differ only in a number (mean over templates)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs, mats = [], []
    for s in args.seeds:
        path = os.path.join(args.runs, f"{args.arm}_vitaminc_s{s}", "last")
        if not os.path.exists(os.path.join(path, "model.pt")):
            continue
        model = SlotEncoder.load(path).to(dev).eval()
        r, D = analyse(model, dev, args.bs)
        r["seed"] = s
        runs.append(r)
        mats.append((s, D))
        print(f"seed {s}: word->digit top-1 {100 * r['digit_word_top1']:.0f}%, six/6 ratio {r['digit_word_ratio']:.2f}, "
              f"last-digit ratio {r['last_digit_ratio']:.2f}, magnitude rho {r['magnitude_spearman']:.2f}, "
              f"same number across templates {r['same_number_other_template_ratio']:.2f}", flush=True)
    out = args.out or f"results/probe_numbers_{args.arm}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(dict(arm=args.arm, templates=TEMPLATES, runs=runs), open(out, "w"), indent=1)
    open(out.replace(".json", ".md"), "w").write(report(args.arm, runs))
    figure(args.arm, mats, out.replace(".json", ".png"))
    print(f"saved {out} and {out.replace('.json', '.md')}")


if __name__ == "__main__":
    main()
