"""
cr_train.py - train the contradiction-only MOLDCAST on a SparseCL benchmark.

The objective is SparseCL's, with MOLDCAST's clash term in place of Hoyer sparsity. For each
training triple (anchor x, its contradiction x+, a similar passage x-), in a batch of B triples:

    logit(x_i, d) = w_K * K(x_i, d) + w_cos * cos(x_i, d),   d in {x+_1..x+_B, x-_1..x-_B}
    loss = - log softmax_d logit(x_i, d)  at d = x+_i

cos comes from the frozen cosine encoder (precomputed once); w_K and w_cos are learned. Passages
that are known contradictions of x_i elsewhere in the data, or identical to x_i, are masked out
of its denominator (false negatives). A hinge keeps each passage's mean slot commitment above a
floor so that slots cannot all fall silent.

Optional objectives for sparse, explainable slots (all off by default):
  --w_sparse, --rho  target sparsity: each slot's mean commitment over the batch is pulled towards
                     rho by KL(rho || mean a_c), so passages commit only to a few slots (holes only
                     where they say something). The commitment floor then drops to rho / 4.
  --w_binary         entropy of each commitment: a_c pushed towards 0 or 1 (crisp holes)
  --w_local          entropy of how a true contradiction's clash is spread over slots: the clash
                     concentrates on one or two slots ("they disagree HERE")
  --w_para           the hard negative is a paraphrase of the anchor: same commitments, same
                     values (squared commitment difference + their per-slot clash)
  --w_balance        different contradictions should use different slots: maximises the entropy
                     of the batch-average clash distribution over slots. Without it, --w_local is
                     satisfied by ONE slot carrying every clash, which explains nothing.
Two degenerate solutions to avoid (seen in tests): --w_sparse alone is satisfied by every slot
committing ~rho for every passage (nothing silent); use --w_binary of the same order as
--w_sparse to make commitments on/off. --w_local alone collapses onto one slot; add --w_balance.
Emergent slot types (no type labels anywhere in training):
  --slot_attention N  competitive slots (Slot Attention, N refinement iterations)
  --minimal_frac f    a fraction f of each batch are MINIMAL-EDIT contradictions generated from
                      training anchors (one fact changed by probe_build.py's rules; the model never
                      sees the edit type), with the anchor's own paraphrase as the hard negative.
                      --w_local then applies only to these rows, where "one slot carries the clash"
                      is the right target; --w_balance spreads them over slots.
Logged: active slots per passage (a > 0.5) and the number of slots carrying 80% of a true
contradiction's clash (smaller = more localised).

    python cr_train.py --dataset hotpotqa --data_dir data --out runs/hotpotqa_s0
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from cr_data import DATASETS, load_triples, triples_file
from cr_model import CosEncoder, SlotEncoder


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="BAAI/bge-base-en-v1.5", help="slot encoder")
    ap.add_argument("--cos_model", default="BAAI/bge-base-en-v1.5", help="frozen cosine encoder")
    ap.add_argument("--n_slots", type=int, default=16)
    ap.add_argument("--value_dim", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_mix", type=int, default=2)
    ap.add_argument("--value_act", choices=["linear", "tanh", "sphere"], default="linear",
                    help="sphere: unit-length values (bounded; helps approximate search)")
    ap.add_argument("--gate_dim", type=int, default=16,
                    help="topic gate on the clash term (0 = ungated K)")
    ap.add_argument("--span_k", type=int, default=0,
                    help="span slots: each slot reads only its top-k tokens (0 = off)")
    ap.add_argument("--span_window", type=int, default=1,
                    help="span slots: neighbours on each side included in a token's local features")
    ap.add_argument("--slot_attention", type=int, default=0,
                    help="Slot Attention head with this many iterations (0 = off)")
    ap.add_argument("--minimal_frac", type=float, default=0.0,
                    help="fraction of each batch made of minimal-edit contradictions")
    ap.add_argument("--minimal_max", type=int, default=30000,
                    help="number of minimal-edit triples generated from training anchors")
    ap.add_argument("--single_pass", action="store_true",
                    help="slot head on the FROZEN cosine encoder: one encoder pass gives both the "
                         "cosine embedding and the slots; only the head is trained")
    ap.add_argument("--init_wk", type=float, default=1.0, help="initial weight of the clash")
    ap.add_argument("--init_wcos", type=float, default=50.0,
                    help="initial weight of cosine (SparseCL's temperature 0.02 = 50)")
    ap.add_argument("--max_len", type=int, default=None, help="default 512 arguana, 256 others")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5, help="encoder")
    ap.add_argument("--head_lr", type=float, default=1e-3, help="slot head and weights")
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--commit_floor", type=float, default=None,
                    help="minimum mean commitment per passage (keeps K from being switched off); "
                         "default 0.3, or rho / 4 with --w_sparse")
    ap.add_argument("--w_sparse", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=0.2, help="target mean commitment per slot")
    ap.add_argument("--w_binary", type=float, default=0.0)
    ap.add_argument("--w_local", type=float, default=0.0)
    ap.add_argument("--w_para", type=float, default=0.0)
    ap.add_argument("--w_balance", type=float, default=0.0)
    ap.add_argument("--w_commit", type=float, default=0.1)
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=None, help="for smoke tests")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="moldcast_v5")
    return ap.parse_args()


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_minimal(train, n_max, seed):
    """minimal-edit triples (anchor, anchor with ONE fact changed, anchor's paraphrase) from the
    training triples' anchors, with probe_build.py's edit rules."""
    import probe_build as pb
    rng = random.Random(seed)
    hard = {}
    for x, _, n in train:
        hard.setdefault(x, n)
    anchors = sorted(hard)
    rng.shuffle(anchors)
    pool = {}
    for t in anchors[:20000]:
        w = t.split()
        for i, j in pb.name_spans(w):
            pool.setdefault(j - i, set()).add(" ".join(pb.split_punct(x)[1] for x in w[i:j]))
    pool = {k: sorted(v) for k, v in pool.items()}
    out, kinds = [], defaultdict(int)
    for x in anchors:
        words = x.split()
        order = list(pb.EDITS)
        rng.shuffle(order)
        for et in order:
            fn = pb.EDITS[et]
            e = fn(words, rng, pool) if et == "entity" else fn(words, rng)
            if e:
                new = pb.apply(words, e)[0]
                if new != words:
                    out.append((x, " ".join(new), hard[x], True))
                    kinds[et] += 1
                    break
        if len(out) >= n_max:
            break
    return out, dict(kinds)


def batch_loss(model, batch, cos_emb, idx, contra, device, amp, args):
    """batch: list of (anchor, pos, neg[, is_minimal]). Returns (loss, stats)."""
    B = len(batch)
    is_min = torch.tensor([len(t) > 3 and bool(t[3]) for t in batch], device=device)
    anchors = [t[0] for t in batch]
    docs = [t[1] for t in batch] + [t[2] for t in batch]
    tok = model.tokenize(anchors + docs, device)
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                        enabled=amp and device.startswith("cuda")):
        a, v = model(tok)
    a, v = a.float(), v.float()
    aq, vq, ad, vd = a[:B], v[:B], a[B:], v[B:]
    cq = cos_emb[torch.tensor([idx[t] for t in anchors], device=device)]
    cd = cos_emb[torch.tensor([idx[t] for t in docs], device=device)]
    K = model.gated_clash(aq, vq, ad, vd, cq, cd)                         # (B, 2B)
    cos = cq @ cd.T
    logits = model.log_wk.exp() * K + model.log_wc.exp() * cos
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for i, x in enumerate(anchors):
        known = contra.get(x, ())
        for j, d in enumerate(docs):
            if j != i and (d == x or d in known):
                mask[i, j] = True
    logits = logits.masked_fill(mask, -1e4)
    labels = torch.arange(B, device=device)
    nce = F.cross_entropy(logits, labels)
    commit = F.relu(args.commit_floor - a.mean(1)).mean()
    loss = nce + args.w_commit * commit
    st = {}
    eps = 1e-6
    # per-slot clash of each true contradiction pair (anchor i, positive i)
    pos_slots = model.clash_per_slot(aq, vq, ad[:B], vd[:B])                # (B, n)
    share = pos_slots / (pos_slots.sum(1, keepdim=True) + eps)
    if args.w_sparse:
        abar = a.mean(0).clamp(eps, 1 - eps)                                # (n,) over passages
        rho = args.rho
        # mean (not sum) over slots, so the term stays on the scale of the contrastive loss
        kl = (rho * torch.log(rho / abar) + (1 - rho) * torch.log((1 - rho) / (1 - abar))).mean()
        loss = loss + args.w_sparse * kl
        st["sparse_kl"] = kl.item()
    if args.w_binary:
        ac = a.clamp(eps, 1 - eps)
        ent = -(ac * ac.log() + (1 - ac) * (1 - ac).log()).mean()
        loss = loss + args.w_binary * ent
        st["binary_ent"] = ent.item()
    if args.w_local:
        ent = -(share * (share + eps).log()).sum(1)
        # with minimal edits in the mix, localisation applies to them only (one fact changed:
        # one slot should carry it); multi-fact contradictions are not forced into one slot
        loc = ent[is_min].mean() if is_min.any() else (ent.mean() if not args.minimal_frac
                                                        else ent.sum() * 0)
        loss = loss + args.w_local * loc
        st["local_ent"] = loc.item()
    if args.w_para:
        a_hn, v_hn = ad[B:], vd[B:]
        para = (aq - a_hn).pow(2).mean() + \
            model.clash_per_slot(aq, vq, a_hn, v_hn).mean()
        loss = loss + args.w_para * para
        st["para"] = para.item()
    if args.w_balance:
        pbar = share.mean(0)                                                # (n,)
        imbalance = math.log(model.n) + (pbar * (pbar + eps).log()).sum()   # log n - entropy >= 0
        loss = loss + args.w_balance * imbalance
        st["imbalance"] = imbalance.item()
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()
        k_pos = K[labels, labels].mean().item()
        k_neg = K[labels, labels + B].mean().item()
        active = (a > 0.5).float().sum(1).mean().item()
        srt = share.sort(1, descending=True).values.cumsum(1)
        slots80 = ((srt < 0.8).sum(1) + 1).float().mean().item()
    if is_min.any():
        with torch.no_grad():
            srt_m = share[is_min].sort(1, descending=True).values.cumsum(1)
            st["minimal_slots_for_80pct"] = ((srt_m < 0.8).sum(1) + 1).float().mean().item()
            st["minimal_acc"] = (logits.argmax(1) == labels)[is_min].float().mean().item()
    st.update(nce=nce.item(), acc=acc, K_pos=k_pos, K_hardneg=k_neg, commit=a.mean().item(),
              active_slots=active, slots_for_80pct_clash=slots80,
              main_slots_used=float(len(set(share.argmax(1).tolist()))),
              masked=mask.float().sum().item() / B)
    return loss, st


@torch.no_grad()
def dev_eval(model, triples, cos_emb, idx, contra, device, amp, args):
    model.eval()
    tot, n = defaultdict(float), 0
    for i in range(0, len(triples) - args.bs + 1, args.bs):
        loss, st = batch_loss(model, triples[i:i + args.bs], cos_emb, idx, contra, device, amp, args)
        for k, v in st.items():
            tot[k] += v
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    args = parse()
    set_seed(args.seed)
    args.max_len = args.max_len or (512 if args.dataset == "arguana" else 256)
    if args.commit_floor is None:
        args.commit_floor = args.rho / 4 if args.w_sparse else 0.3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    os.makedirs(args.out, exist_ok=True)

    train = load_triples(triples_file(args.data_dir, args.dataset, "train"))
    dev = load_triples(triples_file(args.data_dir, args.dataset, "dev"))
    print(f"{args.dataset}: {len(train):,} training triples, {len(dev):,} dev triples")
    minimal = []
    if args.minimal_frac > 0:
        minimal, kinds = make_minimal(train, args.minimal_max, args.seed)
        print(f"minimal-edit triples: {len(minimal):,} from training anchors {kinds}")
        if not minimal:
            raise SystemExit("no minimal edits could be generated from the training anchors")
    contra = defaultdict(set)                       # known contradictions, for false-negative masks
    for x, p, *_ in train + dev + minimal:
        contra[x].add(p)
        contra[p].add(x)

    # frozen cosine embeddings for every text, computed once
    texts = sorted({t for tr in train + dev + minimal for t in tr[:3]})
    idx = {t: i for i, t in enumerate(texts)}
    t0 = time.time()
    cos_enc = CosEncoder(args.cos_model, max_len=args.max_len)
    cos_pooling = cos_enc.pooling
    cos_emb = cos_enc.encode(texts, bs=128, device=device, amp=amp).to(device)
    del cos_enc
    print(f"cosine embeddings for {len(texts):,} texts in {time.time() - t0:.0f}s")

    if args.single_pass:
        args.backbone = args.cos_model                 # the slot head reads the cosine encoder
    model = SlotEncoder(args.backbone, args.n_slots, args.value_dim, args.d_model, 4, args.n_mix,
                        args.max_len, args.value_act, args.gate_dim, cos_emb.size(1),
                        args.init_wk, args.init_wcos, frozen_backbone=args.single_pass,
                        cos_pooling=cos_pooling, span_k=args.span_k,
                        span_window=args.span_window,
                        slot_attention=args.slot_attention).to(device)
    if args.grad_ckpt and not args.single_pass:
        model.encoder.gradient_checkpointing_enable()
    other = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": other, "lr": args.head_lr}]
    if not args.single_pass:
        groups.insert(0, {"params": list(model.encoder.parameters()), "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable parameters: {n_train:,}" + ("  (single pass: encoder frozen)"
                                                  if args.single_pass else ""))
    if len(train) < args.bs:
        raise SystemExit(f"{len(train)} training triples is fewer than one batch of {args.bs}")
    steps_per_epoch = len(train) // args.bs
    total = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = int(args.warmup * total)
    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total - s) / max(total - warm, 1))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=os.path.basename(args.out.rstrip("/")),
                            config=vars(args))
        except Exception as e:                       # noqa: BLE001
            print(f"wandb disabled: {e}")
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    print(f"{total:,} steps ({args.epochs} epochs of {steps_per_epoch:,}), batch {args.bs}")
    model.train()
    step, order, pos = 0, [], 0
    best = math.inf
    while step < total:
        if pos + args.bs > len(order):
            order = list(range(len(train)))
            random.shuffle(order)
            pos = 0
        n_min = int(round(args.minimal_frac * args.bs)) if minimal else 0
        batch = [train[i] for i in order[pos:pos + args.bs - n_min]]
        pos += args.bs - n_min
        if n_min:
            batch += random.sample(minimal, n_min)
        loss, st = batch_loss(model, batch, cos_emb, idx, contra, device, amp, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0 or step == total:
            msg = {f"train/{k}": v for k, v in st.items()}
            msg.update({"train/loss": loss.item(), "w_K": model.log_wk.exp().item(),
                        "w_cos": model.log_wc.exp().item()})
            print(f"step {step}/{total} " + " ".join(f"{k} {v:.4g}" for k, v in msg.items()),
                  flush=True)
            if wb:
                wb.log(msg, step=step)
        if step % steps_per_epoch == 0 or step == total:
            d = dev_eval(model, dev, cos_emb, idx, contra, device, amp, args)
            print(f"dev at step {step}: " + " ".join(f"{k} {v:.4g}" for k, v in d.items()),
                  flush=True)
            if wb:
                wb.log({f"dev/{k}": v for k, v in d.items()}, step=step)
            model.save(os.path.join(args.out, "last"))
            if d.get("nce", math.inf) < best:
                best = d["nce"]
                model.save(os.path.join(args.out, "best"))
    print(f"saved {args.out}/last (and best by dev loss)")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
