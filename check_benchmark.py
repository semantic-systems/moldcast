"""
check_benchmark.py - verify, from the files alone, that the VitaminC benchmark has the properties
its design relies on. Run after build_vitaminc.py; exits with an error if any check fails.

  1. format     both pipelines' loaders read every split (corpus, queries, qrels; triples)
  2. roles      every corpus passage of a split is the target of exactly one query and the
                confounder (the sentence the query paraphrases) of exactly one other
  3. facts      every query keeps the numbers and capitalised words of the sentence it paraphrases
  4. shortcut   for passage-only scores (negation words, length, digits), the share of queries
                whose target outscores its confounder is at most 1/2 (so a score that looks at a
                passage alone cannot solve the benchmark; this is the property SparseCL's
                benchmarks lack)

    python check_benchmark.py --data data
"""
import argparse
import os
import pickle
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cr"))
from build_vitaminc import keeps_facts            # noqa: E402
from cr_data import load_split, load_triples, triples_file   # noqa: E402

NEG = re.compile(r"\b(not|no|never|none|nor|without|n't|neither|cannot)\b", re.I)
FEATURES = {
    "negation words": lambda t: len(NEG.findall(t)),
    "length (words)": lambda t: len(t.split()),
    "digits": lambda t: sum(c.isdigit() for c in t),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    failures = []

    def check(ok, msg):
        print(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    for split in ("train", "dev", "test"):
        print(f"[{split}]")
        corpus, queries, qrels = load_split(args.data, "vitaminc", split)
        with open(os.path.join(args.data, f"vitaminc_{split}_retrieval_gpt4_final.pkl"), "rb") as f:
            own = pickle.load(f)                                # this split's own passages
        check(len(queries) == len(qrels) == len(own) > 0,
              f"format: {len(own)} passages, {len(queries)} queries, {len(qrels)} qrels")
        # roles: target = qrels; confounder = the sentence the query paraphrases (id convention)
        target, confounder = Counter(), Counter()
        pairs = []
        for q, rel in qrels.items():
            (t,) = rel                                          # exactly one relevant passage
            side = q.rsplit("-", 1)[1]
            c = q.replace("-q", "-", 1)                         # split-q7-a -> split-7-a
            pairs.append((q, t, c))
            target[t] += 1
            confounder[c] += 1
            check_ok = c in own and t in own and c != t
            if not check_ok:
                check(False, f"roles: {q} -> target {t}, confounder {c}")
                break
        check(all(target[p] == 1 and confounder[p] == 1 for p in own),
              "roles: every passage is the target of one query and the confounder of one other")
        kept = sum(keeps_facts(corpus[c], queries[q]) for q, t, c in pairs)
        check(kept == len(pairs), f"facts: {kept}/{len(pairs)} queries keep the numbers and names "
                                  "of their sentence")
        for name, f in FEATURES.items():
            wins = sum(f(corpus[t]) > f(corpus[c]) for q, t, c in pairs)
            share = wins / len(pairs)
            check(share <= 0.5 + 1e-9, f"shortcut: '{name}' ranks the target above the "
                                       f"confounder for {100 * share:.1f}% of queries (max 50%)")
        if split in ("train", "dev"):
            tr = load_triples(triples_file(args.data, "vitaminc", split))
            check(len(tr) == len(pairs), f"format: {len(tr)} training triples")
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
