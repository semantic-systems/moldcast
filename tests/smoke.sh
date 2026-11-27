#!/usr/bin/env bash
# tests/smoke.sh - a quick check of the benchmark builder and its validity checks on a small fake
# VitaminC sample: no GPU, no downloads, a few seconds. Run from the repository folder.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d)
python - "$T" <<'PY'
import json, os, random, sys
random.seed(1)
out = os.path.join(sys.argv[1], "src"); os.makedirs(out)
for split, n in (("train", 120), ("validation", 30), ("test", 40)):
    with open(os.path.join(out, f"{split}.jsonl"), "w") as f:
        for i in range(n):
            city, k1, k2 = f"Town{i}", random.randint(10, 99), random.randint(100, 999)
            claim = f"{city} has more than {k1} museums."
            for ev, lab in ((f"{city} has {k1 + 5} museums and {k2} parks.", "SUPPORTS"),
                            (f"{city} has {k1 - 3} museums and {k2} parks.", "REFUTES")):
                f.write(json.dumps(dict(case_id=f"c{i}", claim=claim, evidence=ev, label=lab,
                                        revision_type="real")) + "\n")
PY
python build_vitaminc.py --local_dir "$T/src" --paraphraser test_stub --out "$T/data" --max_pairs 100 20 30 > /dev/null
python check_benchmark.py --data "$T/data" | tail -1
rm -rf "$T"
echo "smoke test passed"
