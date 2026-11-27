"""
interpret_slots.py - what does each slot of a trained MOLDCAST model learn? (VitaminC test split)

The clash decomposes exactly into per-slot terms, a^q_c a^d_c ||v^q_c - v^d_c||^2, so every
contradiction's clash can be attributed to slots. For each trained model (one per seed):

  usage          how often each slot commits (a > 0.5) on the test passages
  localisation   how many slots carry 80% of a contradiction's clash, and the top slot's share
  edit types     each VitaminC pair (a, b) is classified by what its edit changed: a number, a
                 negation, a name (a capitalised word) or another word
  specialisation does the slot that carries the clash (the dominant slot) depend on the edit type?
                 normalised mutual information, against 1,000 shuffles of the edit types
  ablation       switch one slot off (remove its clash term) and measure the drop in pairwise
                 accuracy (target above confounder), overall and per edit type
  attention      the share of the dominant slot's attention that falls on the edited words of the
                 target sentence, against the other slots' share (chance level: the edited words'
                 share of the sentence's tokens)
  examples       for the most important slots, pairs where they carry the clash, edits shown

Writes results/interpret_<arm>.json and a readable report, results/interpret_<arm>.md.

    python interpret_slots.py --arm cr16
"""
import argparse
import difflib
import json
import math
import os
import random
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cr"))
from cr_data import load_split          # noqa: E402
from cr_eval import cached_cos          # noqa: E402
from cr_model import SlotEncoder        # noqa: E402

NEG = {"not", "no", "never", "none", "nor", "without", "n't", "neither", "cannot", "nobody",
       "nothing", "nowhere"}
WORD = re.compile(r"\w+(?:[.,]\w+)*|[^\w\s]")
TYPES = ("number", "negation", "name", "other")


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="cr/runs")
    ap.add_argument("--arm", default="cr16")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--cos_model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--cache_dir", default="cr/cache")
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


# ---------------------------------------------------------------- edits
def words(text):
    return [(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]


def edit(a, b):
    """what changed from a to b: (type, changed words of b, their character spans in b, pretty)."""
    wa, wb = words(a), words(b)
    sm = difflib.SequenceMatcher(a=[w.lower() for w, _, _ in wa], b=[w.lower() for w, _, _ in wb],
                                 autojunk=False)
    ca, cb = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            ca += wa[i1:i2]
            cb += wb[j1:j2]
    changed = [w for w, _, _ in ca + cb]
    if any(any(ch.isdigit() for ch in w) for w in changed):
        kind = "number"
    elif any(w.lower() in NEG for w in changed):
        kind = "negation"
    elif any(w[:1].isupper() and s > 0 for w, s, _ in ca + cb):
        kind = "name"
    else:
        kind = "other"
    pretty = f"{' '.join(w for w, _, _ in ca) or '(nothing)'} -> {' '.join(w for w, _, _ in cb) or '(nothing)'}"
    return kind, [(s, e) for _, s, e in cb], pretty


# ---------------------------------------------------------------- model outputs
@torch.no_grad()
def slots(model, texts, args, dev, spans=None):
    """commitments (N, n), values (N, n, k) and, if spans are given, each slot's attention share on
    those character spans (N, n) with the spans' share of the tokens (N,)."""
    A, V, ATT, BASE = [], [], [], []
    for s in range(0, len(texts), args.bs):
        chunk = texts[s:s + args.bs]
        b = model.tok(chunk, max_length=model.config["max_len"], padding=True, truncation=True,
                      return_tensors="pt", return_offsets_mapping=spans is not None)
        offsets = b.pop("offset_mapping", None)
        b = b.to(dev)
        with torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16, enabled=dev == "cuda"):
            h = model.hidden(b)
            out = model.run_head(h, b, return_attn=spans is not None)
        a, v = out[0].float().cpu(), out[1].float().cpu()
        A.append(a)
        V.append(v)
        if spans is not None:
            attn = out[2].float().cpu()                                   # (B, n, T)
            mask = b["attention_mask"].cpu().bool()
            for i in range(len(chunk)):
                sp = spans[s + i]
                on = torch.tensor([any(st < e2 and en > s2 for s2, e2 in sp) and en > st
                                   for st, en in offsets[i].tolist()]) & mask[i]
                ATT.append(attn[i][:, on].sum(-1) / attn[i][:, mask[i]].sum(-1).clamp_min(1e-9))
                BASE.append(on.float().sum() / mask[i].float().sum())
    res = (torch.cat(A), torch.cat(V))
    if spans is not None:
        res += (torch.stack(ATT), torch.stack(BASE))
    return res


# ---------------------------------------------------------------- statistics
def mutual_info(x, y):
    n = len(x)
    cxy, cx, cy = Counter(zip(x, y)), Counter(x), Counter(y)
    mi = sum(c / n * math.log((c / n) / (cx[a] / n * cy[b] / n)) for (a, b), c in cxy.items())
    hx = -sum(c / n * math.log(c / n) for c in cx.values())
    hy = -sum(c / n * math.log(c / n) for c in cy.values())
    return mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0


def analyse(seed, args, data, dev):
    path = os.path.join(args.runs, f"{args.arm}_vitaminc_s{seed}", "last")
    model = SlotEncoder.load(path).to(dev).eval()
    alpha = json.load(open(os.path.join(path, "alpha_vitaminc_full.json")))["alpha"] * model.k_scale()
    q, t, c, kinds, spans = data["q"], data["t"], data["c"], data["kinds"], data["spans"]
    aq, vq = slots(model, q, args, dev)
    at, vt, att, base = slots(model, t, args, dev, spans=spans)
    ac, vc = slots(model, c, args, dev)
    n = aq.size(1)
    st = model.clash_per_slot(aq, vq, at, vt)                     # (Q, n): target pairs
    sc = model.clash_per_slot(aq, vq, ac, vc)                     # (Q, n): confounder pairs
    cos_t, cos_c = data["cos_t"], data["cos_c"]
    score_t = cos_t + alpha * st.sum(1) / n
    score_c = cos_c + alpha * sc.sum(1) / n
    correct = score_t > score_c
    # usage and localisation
    commit = (torch.cat([at, ac]) > 0.5).float().mean(0)          # (n,)
    share = st / st.sum(1, keepdim=True).clamp_min(1e-12)
    srt = share.sort(1, descending=True).values.cumsum(1)
    slots80 = ((srt < 0.8).sum(1) + 1).float()
    dom = share.argmax(1).tolist()
    # specialisation: dominant slot vs edit type, against shuffled edit types
    nmi = mutual_info(dom, kinds)
    rng = random.Random(seed)
    shuffled = []
    for _ in range(args.shuffles):
        k = kinds[:]
        rng.shuffle(k)
        shuffled.append(mutual_info(dom, k))
    p = (1 + sum(x >= nmi for x in shuffled)) / (1 + len(shuffled))
    # ablation: remove one slot's clash term from both scores
    base_acc = correct.float().mean().item()
    kinds_t = list(kinds)
    abl = []
    for j in range(n):
        ok = (score_t - alpha * st[:, j] / n) > (score_c - alpha * sc[:, j] / n)
        row = dict(overall=base_acc - ok.float().mean().item())
        for kd in TYPES:
            m = torch.tensor([x == kd for x in kinds_t])
            if m.any():
                row[kd] = correct[m].float().mean().item() - ok[m].float().mean().item()
        abl.append(row)
    # attention on the edit: the dominant slot against the other slots
    idx = torch.arange(len(dom))
    d = torch.tensor(dom)
    att_dom = att[idx, d]
    att_other = (att.sum(1) - att_dom) / (n - 1)
    has_edit = base > 0
    per_slot = []
    for j in range(n):
        m = d == j
        types = Counter(kinds[i] for i in m.nonzero().flatten().tolist())
        top_type, top_n = (types.most_common(1)[0] if types else ("-", 0))
        per_slot.append(dict(slot=j, commit=commit[j].item(), dominant_share=m.float().mean().item(),
                             top_type=top_type, top_type_share=top_n / max(int(m.sum()), 1),
                             ablation=abl[j],
                             attention_on_edit=att_dom[m & has_edit].mean().item()
                             if (m & has_edit).any() else None))
    order = sorted(range(n), key=lambda j: -abl[j]["overall"])
    examples = {}
    for j in order[:3]:
        ids = [i for i in range(len(dom)) if dom[i] == j and correct[i]][:args.examples]
        examples[j] = [dict(edit=data["pretty"][i], type=kinds[i], query=q[i], target=t[i],
                            share=share[i, j].item()) for i in ids]
    return dict(seed=seed, slots=n, pairwise=base_acc, commit_rate=commit.mean().item(),
                unused_slots=int((commit < 0.01).sum()), slots_for_80pct=slots80.mean().item(),
                top_slot_share=share.max(1).values.mean().item(), nmi=nmi,
                nmi_shuffled=sum(shuffled) / len(shuffled), nmi_p=p,
                attention_on_edit_dominant=att_dom[has_edit].mean().item(),
                attention_on_edit_other=att_other[has_edit].mean().item(),
                attention_on_edit_chance=base[has_edit].mean().item(),
                per_slot=per_slot, important_slots=order[:3], examples=examples)


# ---------------------------------------------------------------- report
def report(arm, res, kinds):
    L = [f"# What the slots of {arm} learn (VitaminC test, {len(kinds):,} queries)", "",
         "Edit types: " + ", ".join(f"{k} {v:,}" for k, v in Counter(kinds).most_common()), "",
         "## Summary per seed", "",
         "| Seed | Pairwise | Slots used | Slots for 80% of a clash | Top slot's share | Edit-type NMI "
         "(shuffled; p) | Attention on the edit: clash slot / other slots / chance |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in res:
        L.append(f"| {r['seed']} | {100 * r['pairwise']:.1f}% | {r['slots'] - r['unused_slots']} of {r['slots']} "
                 f"| {r['slots_for_80pct']:.1f} | {100 * r['top_slot_share']:.0f}% | {r['nmi']:.3f} "
                 f"({r['nmi_shuffled']:.3f}; p = {r['nmi_p']:.3f}) | {100 * r['attention_on_edit_dominant']:.0f}% / "
                 f"{100 * r['attention_on_edit_other']:.0f}% / {100 * r['attention_on_edit_chance']:.0f}% |")
    for r in res:
        L += ["", f"## Seed {r['seed']}: slot by slot", "",
              "Dominant: share of queries whose clash this slot carries most. Ablation: drop in pairwise accuracy "
              "(points) when the slot is switched off, overall and per edit type.", "",
              "| Slot | Commits | Dominant | Its main edit type | Ablation: overall | number | negation | name | other "
              "| Attention on the edit |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in r["per_slot"]:
            ab = s["ablation"]
            cell = lambda k: f"{100 * ab[k]:.1f}" if k in ab else "-"
            att = f"{100 * s['attention_on_edit']:.0f}%" if s["attention_on_edit"] is not None else "-"
            L.append(f"| {s['slot']} | {100 * s['commit']:.0f}% | {100 * s['dominant_share']:.0f}% | "
                     f"{s['top_type']} ({100 * s['top_type_share']:.0f}%) | {cell('overall')} | {cell('number')} "
                     f"| {cell('negation')} | {cell('name')} | {cell('other')} | {att} |")
        L += ["", "Most important slots (largest ablation drop), with pairs whose clash they carry:"]
        for j in r["important_slots"]:
            L += ["", f"Slot {j}:"]
            for e in r["examples"].get(j, r["examples"].get(str(j), [])):
                L.append(f"- {e['type']}: *{e['edit']}* ({100 * e['share']:.0f}% of the clash). Target: {e['target']}")
    return "\n".join(L) + "\n"


def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus, queries, qrels = load_split(args.data, "vitaminc", "test")
    qids = [x for x in qrels if x in queries]
    tids = [next(iter(qrels[x])) for x in qids]
    cids = [x.replace("-q", "-", 1) for x in qids]
    q = [queries[x] for x in qids]
    t = [corpus[x] for x in tids]
    c = [corpus[x] for x in cids]
    ed = [edit(cc, tt) for cc, tt in zip(c, t)]       # the confounder is the unedited counterpart
    kinds = [e[0] for e in ed]
    Qc = cached_cos(args.cos_model, q, args.max_len, args, dev, dev == "cuda").float().cpu()
    allp = list(corpus)
    Cc = cached_cos(args.cos_model, [corpus[x] for x in allp], args.max_len, args, dev, dev == "cuda").float().cpu()
    pos = {x: i for i, x in enumerate(allp)}
    data = dict(q=q, t=t, c=c, kinds=kinds, spans=[e[1] for e in ed], pretty=[e[2] for e in ed],
                cos_t=(Qc * Cc[[pos[x] for x in tids]]).sum(1), cos_c=(Qc * Cc[[pos[x] for x in cids]]).sum(1))
    res = []
    for s in args.seeds:
        if not os.path.exists(os.path.join(args.runs, f"{args.arm}_vitaminc_s{s}", "last", "model.pt")):
            continue
        r = analyse(s, args, data, dev)
        res.append(r)
        print(f"seed {s}: pairwise {100 * r['pairwise']:.1f}%, slots for 80% {r['slots_for_80pct']:.1f}, "
              f"NMI {r['nmi']:.3f} (shuffled {r['nmi_shuffled']:.3f}, p = {r['nmi_p']:.3f}), attention on "
              f"edit {100 * r['attention_on_edit_dominant']:.0f}% vs {100 * r['attention_on_edit_other']:.0f}% "
              f"(chance {100 * r['attention_on_edit_chance']:.0f}%)", flush=True)
    out = args.out or f"results/interpret_{args.arm}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(dict(arm=args.arm, edit_types=dict(Counter(kinds)), seeds=res), open(out, "w"), indent=1)
    open(out.replace(".json", ".md"), "w").write(report(args.arm, res, kinds))
    print(f"saved {out} and {out.replace('.json', '.md')}")


if __name__ == "__main__":
    main()
