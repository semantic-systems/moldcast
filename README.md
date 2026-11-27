# MOLDCAST: Contradiction Retrieval with a Single Inner Product

This repository contains the code to reproduce every experiment in the paper. It also contains the
code to build the symmetric VitaminC benchmark introduced in the paper.

MOLDCAST is a contradiction detector whose score is a single inner product. Each passage is
described by learned slots. Its score is cosine similarity plus a *clash* term, which can be
rewritten exactly as an inner product between a vector stored once per passage and a vector built
from the query. Any inner-product index can therefore apply it to a whole corpus.

## Contents

```
setup.sh                    one-time setup: SparseCL's environment and code, model and data downloads
download.py                 caches every model and dataset (called by setup.sh)
run_all.sh                  runs the experiments, as SLURM jobs or locally (see below)
make_table.py               prints the paper's tables from the outputs (results/paper_tables.md)

build_vitaminc.py           builds the symmetric VitaminC benchmark (Section 6)
check_benchmark.py          automatic validity checks of the benchmark (Section 6)

cr/cr_model.py              MOLDCAST: slot encoder, clash, inner-product form (Section 3)
cr/cr_train.py              training (Section 3, Appendix A)
cr/cr_eval.py               evaluation: NDCG@10, shortlist reranking and exact search, query-free control
cr/cr_data.py               data loading (SparseCL's file format)
cr/cr_analysis.py           first round: the style probe and the changed-fact test (Section 5)
cr/probe_build.py           first round: sentences with exactly one changed fact (Section 5, Appendix B)
cr/slot_inspect.py          first round: readings of individual slots (Appendix B)
cr/test_cr.py               unit tests of the model and evaluation code

diagnose_ours.py            pairwise accuracy and the test-only corpus for MOLDCAST (Table 3)
diagnose_sparsecl.py        the same for SparseCL, from its own embeddings (Table 3)
make_dump_script.py         writes sparsecl_repo/dump_emb.py: SparseCL's test script that saves embeddings
check_sparsecl_training.py  Hoyer sparsity of SparseCL's trained models (Appendix C)

interpret_slots.py          slot analysis of MOLDCAST 16x8 (Section 8)
explore_slots.py            value clusters of MOLDCAST 16x8 on the corpus (Section 8)
probe_numbers.py            controlled number probe (Section 8, Appendix C)

tests/smoke.sh              a quick check of the benchmark builder (no GPU, a few seconds)
expected_results/           the outputs of our runs behind the paper, to compare yours with (see below)
requirements.txt            the main environment's packages
LICENSE                     MIT (code); the benchmark has its own licence (see Notes)
```

## Requirements

**Hardware.** One GPU per job. All models are bge-base sized (about 110M parameters per encoder),
and training uses gradient checkpointing to reduce memory. On our GPUs,
one MOLDCAST training run on the VitaminC benchmark took about 10 minutes, and one SparseCL run
about 70 minutes. Runs on MSMARCO and HotpotQA take longer, because their training sets are larger.

**Software.** Two Python environments are used.

1. **The main environment**, from which you run everything. It needs Python 3.10 or later and:
   ```
   pip install -r requirements.txt
   pip install matplotlib pytrec_eval      # optional: the probe's figure, and a test cross-check
   ```
2. **SparseCL's environment** (`sparsecl_env/`). SparseCL's code needs older library versions, so
   `setup.sh` creates this environment separately with `uv`, with the exact versions pinned
   (`pip install uv` first).

## Setup

Run once, from this folder, on a machine with internet access (for example a login node):

```
bash setup.sh
```

This script does four things. It creates SparseCL's environment. It clones SparseCL's repository
into `sparsecl_repo/`, at the commit used for the paper (`14fcd8b`). It adds `test_qf.py`, which is
SparseCL's test script with a three-line switch for the query-free control. Finally, it downloads
everything into the Hugging Face cache: bge-base, the two OPUS-MT translation models, VitaminC,
SparseCL's released checkpoints, and SparseCL's benchmark data (into `data/`). After this, all jobs
can run on compute nodes without internet access.

## Running the experiments

Each mode of `run_all.sh` corresponds to a part of the paper. By default, every step is submitted
as a SLURM job, with the correct dependencies between steps. With `LOCAL=1`, the steps run one
after the other on the current machine instead.

| Command | Paper | What it runs |
| --- | --- | --- |
| `bash run_all.sh round1` | Section 5: Tables 1 and 2, Appendix B | zero-shot cosine; MOLDCAST 16x8 and 2x1 on ArguAna, MSMARCO and HotpotQA (3 seeds each), normally and query-free; the style probe and the changed-fact test; SparseCL's released models, normally and query-free; slot readings |
| `bash run_all.sh round2` | Sections 6 and 7: Table 3, Appendix D | builds and checks the VitaminC benchmark; zero-shot cosine; MOLDCAST in six configurations (3 seeds each); SparseCL trained on the benchmark (3 seeds) |
| `bash run_all.sh diagnose` | Table 3, Appendix D | pairwise accuracy and the test-only corpus, for MOLDCAST and SparseCL (after `round2`) |
| `bash run_all.sh validate` | Section 7, Appendix C | SparseCL trained by our pipeline on its own HotpotQA data, against its released model, and the Hoyer check on both benchmarks (after `diagnose`) |
| `bash run_all.sh analysis` | Section 8, Appendix C | slot analysis, value clusters and the number probe of MOLDCAST 16x8 (after `round2`) |
| `bash run_all.sh tables` | all | only rebuilds `results/paper_tables.md` |

`round1` and `round2` are independent and can run at the same time. Every mode skips the steps
that have already finished, so it can safely be run again after a failure. When a mode finishes,
`results/paper_tables.md` contains every table that can be computed from the outputs so far.

**SLURM settings** are environment variables: `GPU` (the GPU option passed to `sbatch`, default
`--gpus-per-node=1`; add a partition here if your cluster needs one, for example
`GPU="--gpus-per-node=1 --partition=gpu"`), `CPUS` (default 8), `TIME_TRAIN` (default 24 hours) and
`TIME_EVAL` (default 12 hours). Each job's script is written to `logs/jobs/`, and its output to
`logs/slurm/` (or `logs/local/` with `LOCAL=1`). The jobs need to find both environments, so start
`run_all.sh` from the activated main environment.

**Outputs.** Trained models and their evaluations are in `cr/runs/<configuration>_<dataset>_s<seed>/last/`.
SparseCL's models and outputs are in `sparsecl_repo/results/` and `sparsecl_repo/test_results/`.
All other results are in `results/`.

## Expected results

`make_table.py` prints the paper's tables. These are the values reported in the paper (mean ±
standard deviation over three seeds where models are trained). Small differences are expected,
because GPU training is not bit-for-bit deterministic, but they should fall within the reported
standard deviations.

**Table 1** (NDCG@10): zero-shot cosine 0.657 / 0.600 / 0.597 (ArguAna / MSMARCO / HotpotQA);
SparseCL 0.704 / 0.909 / 0.966; MOLDCAST 16x8 0.686 / 0.927 / 0.957; MOLDCAST 2x1 0.680 / 0.894 / 0.941.

**Table 2** (share of the gain over cosine kept without the query): MOLDCAST 16x8 85% / 79% / 95%;
SparseCL 79% / 28% / 63%. Style probe AUC of the best single slot: 0.86 / 0.91 / 0.99.

**Table 3** (symmetric VitaminC benchmark): NDCG@10 0.535 (cosine), 0.569 (SparseCL), 0.621
(MOLDCAST 16x8), 0.588 (MOLDCAST 2x1); pairwise accuracy 1.2%, 12.1%, 39.9% and 24.4%; test-only
NDCG@10 0.566, 0.604, 0.668 and 0.629.

**Appendix C**: on HotpotQA, our training of SparseCL reaches NDCG@10 0.965 (released: 0.966), with
median Hoyer sparsity 0.632 for contradictions and 0.283 for confounders (released: 0.631 and 0.281).
On VitaminC, the same training gives 0.209 and 0.210.

**Appendix D** (pairwise accuracy): 2x1 24.4%, 16x1 15.7%, 64x1 16.3%, 2x8 39.6%, 16x8 39.9%,
64x8 35.6%.

**Section 8**: 13 of 16 slots carry 80% of a clash; the largest slot carries 7%. In the number
probe, the same number in another template clashes about 5% as much as a different number, and a
number word finds its digits 25-27% of the time (chance: 1%).

`expected_results/` contains the raw outputs of our runs behind these numbers, in the same format
as yours: the first-round analysis and cosine results, the second-round diagnostics, the validation,
the slot analysis, the value clusters, the number probe, and `paper_tables.md`, the tables
`make_table.py` printed from them. You can compare your `results/` files with them directly.
`expected_results/cr_runs/` holds the evaluation files of every trained MOLDCAST model (NDCG@10 on
dev and test, normally and query-free), in the layout of `cr/runs/`; the model weights are not
included. To print the paper's tables from our outputs without running anything:

```
cp -r expected_results results && mkdir -p cr/runs && cp -r expected_results/cr_runs/* cr/runs/
python make_table.py
```

This reproduces every MOLDCAST number in `paper_tables.md`. SparseCL's NDCG@10 values are read from
the output files of SparseCL's own test script (`sparsecl_repo/test_results/`), which are not
included, so they print as "-" until you run `round1`, `round2` and `validate`.

## Using only the benchmark

To build the symmetric VitaminC benchmark without running any experiment:

```
python build_vitaminc.py --out data       # back-translation: best run on a GPU
python check_benchmark.py --data data
```

The benchmark uses SparseCL's file format (`data/vitaminc_{train,dev,test}_retrieval_gpt4_final.pkl`
and the training triples in `data/vitaminc_{train,dev}_gpt4_final.csv`). The "gpt4" in the file
names only follows SparseCL's naming convention: nothing in the benchmark is generated by GPT-4. Each
pair of contradicting Wikipedia sentences (a, b) from VitaminC enters the corpus. Their
back-translated paraphrases a' and b' are the queries: a' must retrieve b, with a as its confounder,
and b' must retrieve a, with b as its confounder. `data/vitaminc_stats.json` reports the sizes and
shows examples.

## Quick checks

```
bash tests/smoke.sh                       # the benchmark builder and its checks on fake data: seconds, no GPU
```

## Notes

- **Appendix B.** The slot readings in the paper come from pilot models with the same
  configurations as the final ones. `round1` produces the same reports for the final models
  (`cr/runs/cr16_hotpotqa_s0/last/slot_inspect_hotpotqa.md` and the same for `cr2`). The individual
  passages differ from those in the paper, because the slots are learned anew in every run.
- **SparseCL.** All SparseCL results use SparseCL's own code at the pinned commit, with its own
  training settings and test script. The query-free control replaces every query's sparse embedding
  with the average one (the three lines added in `test_qf.py`). The diagnostics of Table 3 apply
  SparseCL's scoring rule to the embeddings that its own script computes (`make_dump_script.py`).
- **Licence.** The code is released under the MIT licence (`LICENSE`). The benchmark is derived
  from VitaminC, which is released under CC BY-SA 3.0 following Wikipedia's terms, so the benchmark
  data built by `build_vitaminc.py` is distributed under CC BY-SA 3.0.
