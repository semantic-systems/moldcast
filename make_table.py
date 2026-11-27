"""
make_table.py - the paper's tables from the outputs of run_all.sh, as Markdown
(python make_table.py > results/paper_tables.md). Results not computed yet show as "-".

  Table 1       first round: SparseCL's benchmarks, NDCG@10 (round1)
  Table 2       first round: the query-free control (round1)
  Section 5     the style probe and the changed-fact test (round1, cr_analysis.py)
  Table 3       second round: the symmetric VitaminC benchmark (round2, diagnose)
  Appendix C    checking our training of SparseCL (validate)
  Appendix D    capacity: six MOLDCAST configurations (round2, diagnose)
  Section 8     slot analysis and the number probe (analysis)

Mean +- standard deviation over three seeds for every trained model.
"""
import json
import re
import statistics

SEEDS = (0, 1, 2)
SCL = "sparsecl_repo/test_results"
ROUND1 = ("arguana", "msmarco", "hotpotqa")
NAMES = {"arguana": "ArguAna", "msmarco": "MSMARCO", "hotpotqa": "HotpotQA"}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def ndcg(path, key="cos_plus_K", part="rerank"):
    r = load(path)
    try:
        return r["results"][part][key]["NDCG@10"]
    except (TypeError, KeyError):
        return None


def moldcast(arm, ds, s, query_free=False, part="rerank"):
    tag = "_queryfree" if query_free else ""
    return ndcg(f"cr/runs/{arm}_{ds}_s{s}/last/eval_{ds}_test{tag}.json", part=part)


def cosine(ds):
    return ndcg(f"results/cos_{ds}_test.json", key="cosine")


def scl_output(path):
    """(SparseCL NDCG@10, its zero-shot cosine NDCG@10) from its test script's output file."""
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return None, None

    def after(marker):
        for i, l in enumerate(lines):
            if l.strip() == marker and i + 1 < len(lines):
                m = re.search(r"'NDCG@10': ([0-9.]+)", lines[i + 1])
                if m:
                    return float(m.group(1))
        return None
    return after("sparsity contradiction retrieval results"), after("cos contradiction retrieval results")


def ms(vals, digits=3, pct=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    k = 100 if pct else 1
    m = statistics.mean(vals) * k
    if len(vals) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {statistics.stdev(vals) * k:.{digits}f}"


def kept(normal, qf, cos):
    """share of the gain over cosine kept without the query, per seed."""
    return [(q - cos) / (n - cos) for n, q in zip(normal, qf)
            if None not in (n, q, cos) and n - cos > 1e-6]


def main():
    L = ["# Results of the paper (from run_all.sh outputs)", ""]

    # ---- Table 1: first round
    L += ["## Table 1. First round: SparseCL's benchmarks (NDCG@10, test)", "",
          "| System | ArguAna | MSMARCO | HotpotQA | Numbers |", "| --- | --- | --- | --- | --- |",
          "| Zero-shot cosine | " + " | ".join(ms([cosine(d)]) for d in ROUND1) + " | 768 |",
          "| SparseCL (released bge models) | " + " | ".join(
              ms([scl_output(f"{SCL}/test_{d}_bge_zeroshot_hoyer/{d}_parallel_output")[0]]) for d in ROUND1) + " | 1,536 |"]
    for arm, label, numbers in (("cr16", "MOLDCAST, 16 x 8", "928"), ("cr2", "MOLDCAST, 2 x 1", "774")):
        L.append(f"| {label} | " + " | ".join(ms([moldcast(arm, d, s) for s in SEEDS]) for d in ROUND1) + f" | {numbers} |")
    L += ["", "Exact search over the whole corpus, MOLDCAST 16 x 8: " + ", ".join(
        f"{NAMES[d]} {ms([moldcast('cr16', d, s, part='full') for s in SEEDS])}" for d in ROUND1)]

    # ---- Table 2: query-free control
    L += ["", "## Table 2. The query-free control on SparseCL's benchmarks (NDCG@10, test)", "",
          "| System | ArguAna | MSMARCO | HotpotQA |", "| --- | --- | --- | --- |"]
    sc = {d: scl_output(f"{SCL}/test_{d}_bge_zeroshot_hoyer/{d}_parallel_output") for d in ROUND1}
    sq = {d: scl_output(f"{SCL}/test_{d}_bge_zeroshot_hoyer_qf/{d}_parallel_output") for d in ROUND1}
    L.append("| SparseCL | " + " | ".join(ms([sc[d][0]]) for d in ROUND1) + " |")
    L.append("| query-free | " + " | ".join(ms([sq[d][0]]) for d in ROUND1) + " |")
    L.append("| gain kept (%) | " + " | ".join(
        ms(kept([sc[d][0]], [sq[d][0]], sc[d][1] if sc[d][1] is not None else cosine(d)), 0, pct=True) for d in ROUND1) + " |")
    n = {d: [moldcast("cr16", d, s) for s in SEEDS] for d in ROUND1}
    q = {d: [moldcast("cr16", d, s, True) for s in SEEDS] for d in ROUND1}
    L.append("| MOLDCAST, 16 x 8 | " + " | ".join(ms(n[d]) for d in ROUND1) + " |")
    L.append("| query-free | " + " | ".join(ms(q[d]) for d in ROUND1) + " |")
    L.append("| gain kept (%) | " + " | ".join(ms(kept(n[d], q[d], cosine(d)), 0, pct=True) for d in ROUND1) + " |")

    # ---- Section 5: style probe and changed-fact test
    an = {d: [load(f"results/analysis_cr16_{d}_s{s}.json") or {} for s in SEEDS] for d in ROUND1}
    get = lambda d, k: [a.get(k) for a in an[d]]
    L += ["", "## Section 5. What MOLDCAST 16 x 8 reads from a passage alone (cr_analysis.py)", "",
          "| Measure | ArguAna | MSMARCO | HotpotQA |", "| --- | --- | --- | --- |",
          "| Style probe: best single slot, AUC | " + " | ".join(ms(get(d, "style_best_slot_auc")) for d in ROUND1) + " |",
          "| Changed fact vs deleted article: clash ratio | " + " | ".join(ms(get(d, "sensitivity_ratio"), 1) for d in ROUND1) + " |"]

    # ---- Table 3: second round
    dm = load("results/diagnose_ours.json") or {}
    dsc = load("results/diagnose_sparsecl.json") or {}
    sv = [scl_output(f"{SCL}/test_vitaminc_bge_sparsity_s{s}/vitaminc_parallel_output")[0] for s in SEEDS]
    svq = [scl_output(f"{SCL}/test_vitaminc_bge_sparsity_s{s}_qf/vitaminc_parallel_output")[0] for s in SEEDS]

    def diag(src, prefix, tag, field):
        return [src.get(f"{prefix}_s{s}{tag}", {}).get(field) for s in SEEDS]
    L += ["", "## Table 3. Second round: the symmetric benchmark (test)", "",
          "| System | NDCG@10 | NDCG@10, query-free | Pairwise (%) | Pairwise, query-free (%) "
          "| Test-only NDCG@10 | Test-only, query-free |", "| --- | --- | --- | --- | --- | --- | --- |"]
    c = dm.get("cosine", {})
    L.append(f"| Zero-shot cosine | {ms([cosine('vitaminc')])} | - | {ms([c.get('pairwise')], 1, pct=True)} | - "
             f"| {ms([c.get('test_only_ndcg10')])} | - |")
    L.append(f"| SparseCL | {ms(sv)} | {ms(svq)} | {ms(diag(dsc, 'sparsecl', '', 'pairwise'), 1, pct=True)} "
             f"| {ms(diag(dsc, 'sparsecl', '_queryfree', 'pairwise'), 1, pct=True)} "
             f"| {ms(diag(dsc, 'sparsecl', '', 'test_only_ndcg10'))} | {ms(diag(dsc, 'sparsecl', '_queryfree', 'test_only_ndcg10'))} |")
    for arm, label in (("cr16", "MOLDCAST, 16 x 8"), ("cr2", "MOLDCAST, 2 x 1")):
        L.append(f"| {label} | {ms([moldcast(arm, 'vitaminc', s) for s in SEEDS])} "
                 f"| {ms([moldcast(arm, 'vitaminc', s, True) for s in SEEDS])} "
                 f"| {ms(diag(dm, arm, '', 'pairwise'), 1, pct=True)} | {ms(diag(dm, arm, '_queryfree', 'pairwise'), 1, pct=True)} "
                 f"| {ms(diag(dm, arm, '', 'test_only_ndcg10'))} | {ms(diag(dm, arm, '_queryfree', 'test_only_ndcg10'))} |")

    # ---- Appendix C: validation of our SparseCL training
    vh, vv = load("results/validate_hotpotqa.json") or {}, load("results/validate_vitaminc.json") or {}
    rel, our = vh.get("results/val_emb_released.npz", {}), vh.get("results/val_emb_ours.npz", {})
    vit = list(vv.values())
    L += ["", "## Appendix C. Checking our training of SparseCL", "",
          "| Data | Model | NDCG@10 | Hoyer: contradiction | Hoyer: confounder | Sparse score prefers the contradiction (%) |",
          "| --- | --- | --- | --- | --- | --- |",
          f"| HotpotQA | released | {ms([scl_output(f'{SCL}/test_hotpotqa_bge_zeroshot_hoyer/hotpotqa_parallel_output')[0]])} "
          f"| {ms([rel.get('target')])} | {ms([rel.get('confounder')])} | {ms([rel.get('pairwise')], 1, pct=True)} |",
          f"| HotpotQA | our training | {ms([scl_output(f'{SCL}/test_hotpotqa_bge_sparsity_val/hotpotqa_parallel_output')[0]])} "
          f"| {ms([our.get('target')])} | {ms([our.get('confounder')])} | {ms([our.get('pairwise')], 1, pct=True)} |",
          f"| VitaminC | our training | {ms(sv)} | {ms([x.get('target') for x in vit])} "
          f"| {ms([x.get('confounder') for x in vit])} | {ms([x.get('pairwise') for x in vit], 1, pct=True)} |"]

    # ---- Appendix D: capacity
    L += ["", "## Appendix D. How much capacity MOLDCAST needs (symmetric benchmark, test)", "",
          "| Configuration | Numbers | NDCG@10 | Pairwise (%) | Pairwise, query-free (%) |", "| --- | --- | --- | --- | --- |"]
    for arm, label, numbers in (("cr2", "2 x 1", "774"), ("cr16x1", "16 x 1", "816"), ("cr64x1", "64 x 1", "960"),
                                ("cr2x8", "2 x 8", "788"), ("cr16", "16 x 8", "928"), ("cr64x8", "64 x 8", "1,408")):
        L.append(f"| MOLDCAST, {label} | {numbers} | {ms([moldcast(arm, 'vitaminc', s) for s in SEEDS])} "
                 f"| {ms(diag(dm, arm, '', 'pairwise'), 1, pct=True)} | {ms(diag(dm, arm, '_queryfree', 'pairwise'), 1, pct=True)} |")
    L.append(f"| SparseCL | 1,536 | {ms(sv)} | {ms(diag(dsc, 'sparsecl', '', 'pairwise'), 1, pct=True)} "
             f"| {ms(diag(dsc, 'sparsecl', '_queryfree', 'pairwise'), 1, pct=True)} |")

    # ---- Section 8: slot analysis and number probe
    it = (load("results/interpret_cr16.json") or {}).get("seeds", [])
    L += ["", "## Section 8. What the 16 slots of MOLDCAST 16 x 8 do (per seed; full report: results/interpret_cr16.md)", "",
          "| Seed | Slots for 80% of a clash | Top slot's share | Edit-type NMI (shuffled) | Attention on edit / chance | Ablation per slot (points) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in it:
        drops = [s["ablation"]["overall"] for s in r["per_slot"]]
        L.append(f"| {r['seed']} | {r['slots_for_80pct']:.1f} | {100 * r['top_slot_share']:.0f}% | {r['nmi']:.3f} "
                 f"({r['nmi_shuffled']:.3f}) | {100 * r['attention_on_edit_dominant']:.0f}% / {100 * r['attention_on_edit_chance']:.0f}% "
                 f"| {100 * min(drops):.1f}-{100 * max(drops):.1f} |")
    pr = (load("results/probe_numbers_cr16.json") or {}).get("runs", [])
    if not pr:                                  # not computed yet: one column of "-"
        pr = [{"seed": "-", **{k: None for k in ("same_number_other_template_ratio", "digit_word_top1",
                                                  "digit_word_ratio", "last_digit_ratio", "magnitude_spearman")}}]
    L += ["", "## Section 8 / Appendix C table. How MOLDCAST 16 x 8 encodes numbers (probe_numbers.py)", "",
          "| Measure | " + " | ".join(f"seed {r['seed']}" for r in pr) + " |", "| --- |" + " --- |" * len(pr)]
    for key, label, pct in (("same_number_other_template_ratio", "Same number, other template (clash ratio)", False),
                            ("digit_word_top1", "Word finds its digit (top-1; chance 1%)", True),
                            ("digit_word_ratio", "\"six\" against \"6\" (clash ratio)", False),
                            ("last_digit_ratio", "Same last digit, e.g. 6 and 16 (clash ratio)", False),
                            ("magnitude_spearman", "Clash against distance in size (Spearman)", False)):
        L.append(f"| {label} | " + " | ".join("-" if r[key] is None else (f"{100 * r[key]:.0f}%" if pct else f"{r[key]:.2f}")
                                            for r in pr) + " |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
