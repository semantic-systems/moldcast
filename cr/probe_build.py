"""
probe_build.py - single-edit contradiction pairs with a KNOWN changed span, for testing explanations.

Each pair is a passage from the benchmark corpus and a copy with exactly ONE fact changed by a rule,
so the contradiction lives in a known span (word positions recorded on both sides). Edit types:

  year      1993 -> 2011            (a year shifted by 3-40 years)
  number    7,500 -> 12,000         (a non-year number changed)
  month     October -> March
  negation  "was released" <-> "was not released"
  antonym   won <-> lost, first <-> last, increased <-> decreased, ...
  entity    "Jon Turteltaub" -> a capitalised name of the same length from another passage

Rule-based on purpose (no NLP library needed). Edits are word-level: passages are split on
whitespace, the edited words are replaced, and the changed word positions are stored for both the
original and the edited text.

    python probe_build.py --dataset hotpotqa --per_type 150 --out data/probe_hotpotqa.jsonl
"""
import argparse
import json
import random
import re
from collections import Counter

from cr_data import load_split

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
AUX = {"is", "was", "are", "were", "has", "have", "had", "does", "did", "do", "can", "could",
       "will", "would", "should"}
ANTONYMS = [("won", "lost"), ("first", "last"), ("largest", "smallest"), ("increased", "decreased"),
            ("before", "after"), ("positive", "negative"), ("success", "failure"),
            ("north", "south"), ("east", "west"), ("older", "younger"), ("higher", "lower"),
            ("earliest", "latest"), ("best", "worst"), ("male", "female"), ("highest", "lowest"),
            ("successful", "unsuccessful"), ("public", "private"), ("early", "late"),
            ("larger", "smaller"), ("more", "fewer"), ("rise", "fall"), ("winner", "loser")]
ANT = {a: b for a, b in ANTONYMS} | {b: a for a, b in ANTONYMS}
NOT_NAMES = {"The", "A", "An", "In", "On", "At", "It", "He", "She", "They", "This", "That", "These",
             "Those", "His", "Her", "Their", "Its", "As", "By", "For", "From", "With", "And", "But",
             "Of", "To", "After", "Before", "During", "While", "When", "Since", "However", "Also",
             "I", "We", "You"} | set(MONTHS)

YEAR = re.compile(r"^(\(?)(1[6-9]\d\d|20[0-2]\d)([).,;:'\"]*)$")
NUMBER = re.compile(r"^(\(?\$?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?([).,;:%'\"]*)$")
PUNCT = re.compile(r"^([(\"']*)(.*?)([).,;:!?'\"]*)$")


def split_punct(w):
    m = PUNCT.match(w)
    return m.group(1), m.group(2), m.group(3)


def edit_year(words, rng):
    cands = [i for i, w in enumerate(words) if YEAR.match(w)]
    if not cands:
        return None
    i = rng.choice(cands)
    pre, y, post = YEAR.match(words[i]).groups()
    shift = rng.choice([s for s in range(-40, 41) if abs(s) >= 3])
    ny = min(max(int(y) + shift, 1600), 2025)
    if ny == int(y):
        return None
    return [(i, i + 1, [f"{pre}{ny}{post}"])]


def edit_number(words, rng):
    cands = [i for i, w in enumerate(words) if NUMBER.match(w) and not YEAR.match(w)]
    if not cands:
        return None
    i = rng.choice(cands)
    pre, num, frac, post = NUMBER.match(words[i]).groups()
    v = int(num.replace(",", ""))
    nv = v + rng.choice([-1, 1]) * max(1, int(v * rng.uniform(0.3, 0.9)))
    if nv <= 0 or nv == v:
        nv = v * 2 + 3
    s = f"{nv:,}" if "," in num else str(nv)
    return [(i, i + 1, [f"{pre}{s}{frac or ''}{post}"])]


def edit_month(words, rng):
    cands = [i for i, w in enumerate(words) if split_punct(w)[1] in MONTHS]
    if not cands:
        return None
    i = rng.choice(cands)
    pre, m, post = split_punct(words[i])
    return [(i, i + 1, [pre + rng.choice([x for x in MONTHS if x != m]) + post])]


def edit_negation(words, rng):
    cands = [i for i in range(len(words) - 1) if words[i].lower() in AUX]
    if not cands:
        return None
    i = rng.choice(cands)
    if words[i + 1].lower() == "not":                     # remove an existing negation
        return [(i, i + 2, [words[i]])]
    return [(i, i + 1, [words[i], "not"])]                 # insert one


def edit_antonym(words, rng):
    cands = [i for i, w in enumerate(words) if split_punct(w)[1].lower() in ANT]
    if not cands:
        return None
    i = rng.choice(cands)
    pre, core, post = split_punct(words[i])
    new = ANT[core.lower()]
    if core[:1].isupper():
        new = new.capitalize()
    return [(i, i + 1, [pre + new + post])]


def name_spans(words):
    """maximal runs of capitalised words that are not sentence-initial or function words."""
    spans, i = [], 0
    while i < len(words):
        core = split_punct(words[i])[1]
        start_of_sentence = i == 0 or words[i - 1][-1:] in ".!?"
        if core[:1].isupper() and core not in NOT_NAMES and not start_of_sentence \
                and core.isalpha():
            j = i
            while j + 1 < len(words) and split_punct(words[j + 1])[1][:1].isupper() \
                    and split_punct(words[j + 1])[1].isalpha() \
                    and split_punct(words[j])[2] == "":
                j += 1
            spans.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return spans


def edit_entity(words, rng, pool):
    spans = name_spans(words)
    if not spans:
        return None
    i, j = rng.choice(spans)
    old = " ".join(split_punct(w)[1] for w in words[i:j])
    same_len = [p for p in pool.get(j - i, []) if p != old]
    if not same_len:
        return None
    new = rng.choice(same_len).split()
    post = split_punct(words[j - 1])[2]
    pre = split_punct(words[i])[0]
    new[0] = pre + new[0]
    new[-1] = new[-1] + post
    return [(i, j, new)]


EDITS = {"year": edit_year, "number": edit_number, "month": edit_month,
         "negation": edit_negation, "antonym": edit_antonym, "entity": edit_entity}


def apply(words, edit):
    """one (i, j, new_words) edit -> edited words, changed positions in original and edited."""
    (i, j, new), = edit
    out = words[:i] + new + words[j:]
    return out, list(range(i, j)), list(range(i, i + len(new)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per_type", type=int, default=150)
    ap.add_argument("--min_words", type=int, default=15)
    ap.add_argument("--max_words", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    corpus, queries, qrels = load_split(args.data_dir, args.dataset, args.split)
    # source passages: the corpus passages of this split's queries (paraphrases in the synthetic
    # sets, arguments in ArguAna), restricted to a length that keeps occlusion affordable
    texts = sorted({queries[q] for q in qrels if q in queries})
    texts = [t for t in texts if args.min_words <= len(t.split()) <= args.max_words]
    rng.shuffle(texts)
    pool = {}
    for t in texts[:20000]:
        w = t.split()
        for i, j in name_spans(w):
            pool.setdefault(j - i, set()).add(" ".join(split_punct(x)[1] for x in w[i:j]))
    pool = {k: sorted(v) for k, v in pool.items()}
    out, count = [], Counter()
    order = list(EDITS)
    for t in texts:
        if all(count[e] >= args.per_type for e in order):
            break
        words = t.split()
        rng.shuffle(order)
        for etype in order:
            if count[etype] >= args.per_type:
                continue
            fn = EDITS[etype]
            edit = fn(words, rng, pool) if etype == "entity" else fn(words, rng)
            if not edit:
                continue
            new_words, span_o, span_e = apply(words, edit)
            if new_words == words:
                continue
            out.append(dict(type=etype, original=" ".join(words), edited=" ".join(new_words),
                            span_original=span_o, span_edited=span_e))
            count[etype] += 1
            break                                   # one edit per source passage
    path = args.out or f"{args.data_dir}/probe_{args.dataset}.jsonl"
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} single-edit pairs from {args.dataset} {args.split}: {dict(count)}")
    for r in out[:6]:
        o, e = r["original"].split(), r["edited"].split()
        print(f"  [{r['type']}] {' '.join(o[i] for i in r['span_original'])!r} -> "
              f"{' '.join(e[i] for i in r['span_edited'])!r}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
