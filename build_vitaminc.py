"""
build_vitaminc.py - a contradiction-retrieval benchmark on which a passage alone tells nothing.

Source: VitaminC (Schuster et al., 2021), real Wikipedia revisions. For one claim, a sentence
before an edit and the same sentence after it carry opposite verdicts (one SUPPORTS the claim,
the other REFUTES it): two near-identical, human-written sentences that contradict each other.

Construction, for every such pair (a, b):
  corpus   a and b, the original Wikipedia sentences (no generated text in the corpus)
  queries  a' = a paraphrase of a, whose relevant passage is b (and a is its confounder)
           b' = a paraphrase of b, whose relevant passage is a (and b is its confounder)
Every corpus passage is the target of exactly one query and the confounder of exactly one other.
A score that looks at a passage alone must prefer b over a to answer a', and a over b to answer
b': whatever it is, it answers at most one of the two (at most half of all queries). Cosine is
challenged as in SparseCL's benchmarks: a' paraphrases a, so cosine tends to rank the confounder
a above the contradiction b.

Paraphrases come from back-translation (English -> German -> English, MarianMT) and are kept
only when they preserve every number and every capitalised word of the source (the facts that
VitaminC edits change); a pair is kept only if both of its sentences get a paraphrase.

Output, in SparseCL's file format (read unchanged by SparseCL's test script and by cr/cr_data.py;
"gpt4" in the names is that format's naming convention, nothing here comes from GPT-4):
  data/vitaminc_{train,dev,test}_retrieval_gpt4_final.pkl   corpus, queries, qrels (3 pickles)
  data/vitaminc_{train,dev}_gpt4_final.csv                 training triples
                                                           (query, contradiction, confounder)
  data/vitaminc_stats.json                                  counts and 40 examples to read

    python build_vitaminc.py --out data                     # on a GPU node (downloads models)
"""
import argparse
import csv
import json
import os
import pickle
import random
import re
from collections import Counter, defaultdict

SPLITS = (("train", "train"), ("dev", "validation"), ("test", "test"))
NUM = re.compile(r"\d+(?:[.,]\d+)*")
CAP = re.compile(r"\b[A-Z][\w'\-]+")


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--max_pairs", nargs=3, type=int, default=[20000, 1000, 2000],
                    metavar=("TRAIN", "DEV", "TEST"))
    ap.add_argument("--include_synthetic", action="store_true",
                    help="also use VitaminC's synthetic revisions (default: real ones only)")
    ap.add_argument("--local_dir", default=None,
                    help="read {train,validation,test}.jsonl from here instead of the hub")
    ap.add_argument("--paraphraser", default="backtranslate",
                    choices=["backtranslate", "test_stub"],
                    help="test_stub: a trivial rewrite for testing the pipeline only")
    ap.add_argument("--mt_fwd", default="Helsinki-NLP/opus-mt-en-de")
    ap.add_argument("--mt_bwd", default="Helsinki-NLP/opus-mt-de-en")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------- VitaminC
def load_split(name, args):
    if args.local_dir:
        with open(os.path.join(args.local_dir, f"{name}.jsonl")) as f:
            rows = [json.loads(l) for l in f]
    else:
        from datasets import load_dataset
        rows = list(load_dataset("tals/vitaminc", split=name))
    need = {"claim", "evidence", "label"}
    if rows and not need <= set(rows[0]):
        raise SystemExit(f"VitaminC {name}: expected columns {sorted(need)}, found {sorted(rows[0])}")
    return rows


def label_of(v):
    if isinstance(v, int):
        return {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}.get(v, str(v))
    return str(v).strip().upper()


def norm(t):
    return " ".join(t.split())


def contradiction_pairs(rows, args, used):
    """(a, b): two evidence sentences, one SUPPORTS and one REFUTES the same claim."""
    by_claim = defaultdict(lambda: {"SUPPORTS": set(), "REFUTES": set()})
    for r in rows:
        if not args.include_synthetic and str(r.get("revision_type", "real")).lower() != "real":
            continue
        lab = label_of(r["label"])
        if lab in ("SUPPORTS", "REFUTES"):
            key = (r.get("case_id", ""), norm(r["claim"]).lower())
            by_claim[key][lab].add(norm(r["evidence"]))
    pairs = []
    for key in sorted(by_claim):
        sup, ref = sorted(by_claim[key]["SUPPORTS"]), sorted(by_claim[key]["REFUTES"])
        for s in sup:
            for t in ref:
                if s != t and s not in used and t not in used:   # each sentence in one pair only
                    pairs.append((s, t))
                    used.update((s, t))
                    break
            else:
                continue
            break
    return pairs


# ---------------------------------------------------------------- paraphrases
class BackTranslator:
    def __init__(self, args):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.m = []
        for name in (args.mt_fwd, args.mt_bwd):
            tok = MarianTokenizer.from_pretrained(name)
            mod = MarianMTModel.from_pretrained(name).to(self.dev).eval()
            if self.dev == "cuda":
                mod = mod.half()
            self.m.append((tok, mod))

    def _translate(self, texts, k):
        tok, mod = self.m[k]
        out = []
        for s in range(0, len(texts), self.args.bs):
            b = tok(texts[s:s + self.args.bs], return_tensors="pt", padding=True, truncation=True,
                    max_length=256).to(self.dev)
            with self.torch.no_grad():
                g = mod.generate(**b, num_beams=self.args.beams, max_new_tokens=256)
            out += tok.batch_decode(g, skip_special_tokens=True)
        return out

    def __call__(self, texts):
        return self._translate(self._translate(texts, 0), 1)


def stub_paraphrase(texts):
    """for tests only: a meaning-preserving rewrite that keeps every number and name."""
    return [("According to the article, " + t[0].lower() + t[1:]) if t else t for t in texts]


def keeps_facts(src, para):
    """the paraphrase keeps every number and capitalised word (not the first word) and differs."""
    if not para or norm(para).lower() == norm(src).lower():
        return False
    if Counter(NUM.findall(src)) != Counter(NUM.findall(para)):
        return False
    caps = lambda t: {w for i, w in enumerate(CAP.findall(t)) if not (i == 0 and t.startswith(w))}
    return caps(src) <= set(CAP.findall(para))


# ---------------------------------------------------------------- write
def write(split, pairs, paras, args):
    corpus, queries, qrels, triples = {}, {}, {}, []
    for i, (a, b) in enumerate(pairs):
        ia, ib = f"{split}-{i}-a", f"{split}-{i}-b"
        corpus[ia], corpus[ib] = {"text": a}, {"text": b}
        qa, qb = f"{split}-q{i}-a", f"{split}-q{i}-b"
        queries[qa], queries[qb] = paras[a], paras[b]
        qrels[qa], qrels[qb] = {ib: 1}, {ia: 1}
        triples += [(paras[a], b, a), (paras[b], a, b)]          # (query, contradiction, confounder)
    with open(os.path.join(args.out, f"vitaminc_{split}_retrieval_gpt4_final.pkl"), "wb") as f:
        pickle.dump(corpus, f)
        pickle.dump(queries, f)
        pickle.dump(qrels, f)
    if split in ("train", "dev"):
        with open(os.path.join(args.out, f"vitaminc_{split}_gpt4_final.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sent0", "sent1", "hard_neg"])
            w.writerows(triples)
    return dict(pairs=len(pairs), corpus=len(corpus), queries=len(queries))


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    para_fn = BackTranslator(args) if args.paraphraser == "backtranslate" else stub_paraphrase
    stats, used, examples = {}, set(), []
    for (split, hub_name), cap in zip(SPLITS, args.max_pairs):
        pairs = contradiction_pairs(load_split(hub_name, args), args, used)
        rng.shuffle(pairs)
        pairs = pairs[:int(cap * 1.6) + 10]                      # headroom for the paraphrase filter
        texts = sorted({t for p in pairs for t in p})
        paras = dict(zip(texts, para_fn(texts)))
        ok = {t for t in texts if keeps_facts(t, paras[t])}
        kept = [p for p in pairs if p[0] in ok and p[1] in ok][:cap]
        stats[split] = write(split, kept, paras, args)
        stats[split]["candidate_pairs"] = len(pairs)
        stats[split]["paraphrases_kept"] = f"{len(ok)}/{len(texts)}"
        examples += [dict(split=split, a=a, b=b, query_a=paras[a], query_b=paras[b])
                     for a, b in kept[:40 // len(SPLITS) + 1]]
        print(f"{split}: {stats[split]}", flush=True)
    with open(os.path.join(args.out, "vitaminc_stats.json"), "w") as f:
        json.dump(dict(args=vars(args), splits=stats, examples=examples[:40]), f, indent=1)
    print(f"wrote {args.out}/vitaminc_* (read the examples in vitaminc_stats.json)")


if __name__ == "__main__":
    main()
