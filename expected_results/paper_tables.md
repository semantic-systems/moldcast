# Results of the paper (from run_all.sh outputs)

## Table 1. First round: SparseCL's benchmarks (NDCG@10, test)

| System | ArguAna | MSMARCO | HotpotQA | Numbers |
| --- | --- | --- | --- | --- |
| Zero-shot cosine | 0.657 | 0.600 | 0.597 | 768 |
| SparseCL (released bge models) | 0.704 | 0.909 | 0.966 | 1,536 |
| MOLDCAST, 16 x 8 | 0.686 ± 0.002 | 0.927 ± 0.001 | 0.957 ± 0.001 | 928 |
| MOLDCAST, 2 x 1 | 0.680 ± 0.002 | 0.894 ± 0.000 | 0.941 ± 0.003 | 774 |

Exact search over the whole corpus, MOLDCAST 16 x 8: ArguAna 0.686 ± 0.002, MSMARCO 0.927 ± 0.001, HotpotQA 0.957 ± 0.001

## Table 2. The query-free control on SparseCL's benchmarks (NDCG@10, test)

| System | ArguAna | MSMARCO | HotpotQA |
| --- | --- | --- | --- |
| SparseCL | 0.704 | 0.909 | 0.966 |
| query-free | 0.695 | 0.686 | 0.827 |
| gain kept (%) | 79 | 28 | 63 |
| MOLDCAST, 16 x 8 | 0.686 ± 0.002 | 0.927 ± 0.001 | 0.957 ± 0.001 |
| query-free | 0.682 ± 0.003 | 0.859 ± 0.004 | 0.940 ± 0.001 |
| gain kept (%) | 85 ± 8 | 79 ± 1 | 95 ± 0 |

## Section 5. What MOLDCAST 16 x 8 reads from a passage alone (cr_analysis.py)

| Measure | ArguAna | MSMARCO | HotpotQA |
| --- | --- | --- | --- |
| Style probe: best single slot, AUC | 0.861 ± 0.011 | 0.908 ± 0.004 | 0.989 ± 0.001 |
| Changed fact vs deleted article: clash ratio | 6.8 ± 1.4 | 82.6 ± 9.0 | 108.8 ± 10.9 |

## Table 3. Second round: the symmetric benchmark (test)

| System | NDCG@10 | NDCG@10, query-free | Pairwise (%) | Pairwise, query-free (%) | Test-only NDCG@10 | Test-only, query-free |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot cosine | 0.535 | - | 1.2 | - | 0.566 | - |
| SparseCL | 0.569 ± 0.002 | 0.535 ± 0.001 | 12.1 ± 1.3 | 1.4 ± 0.2 | 0.604 ± 0.003 | 0.559 ± 0.008 |
| MOLDCAST, 16 x 8 | 0.621 ± 0.002 | 0.559 ± 0.001 | 39.9 ± 1.2 | 12.5 ± 1.4 | 0.668 ± 0.001 | 0.595 ± 0.002 |
| MOLDCAST, 2 x 1 | 0.588 ± 0.024 | 0.545 ± 0.008 | 24.4 ± 8.1 | 8.6 ± 4.8 | 0.629 ± 0.025 | 0.581 ± 0.012 |

## Appendix C. Checking our training of SparseCL

| Data | Model | NDCG@10 | Hoyer: contradiction | Hoyer: confounder | Sparse score prefers the contradiction (%) |
| --- | --- | --- | --- | --- | --- |
| HotpotQA | released | 0.966 | 0.631 | 0.281 | 96.8 |
| HotpotQA | our training | 0.965 | 0.632 | 0.283 | 96.8 |
| VitaminC | our training | 0.569 ± 0.002 | 0.209 ± 0.001 | 0.210 ± 0.002 | 44.2 ± 0.2 |

## Appendix D. How much capacity MOLDCAST needs (symmetric benchmark, test)

| Configuration | Numbers | NDCG@10 | Pairwise (%) | Pairwise, query-free (%) |
| --- | --- | --- | --- | --- |
| MOLDCAST, 2 x 1 | 774 | 0.588 ± 0.024 | 24.4 ± 8.1 | 8.6 ± 4.8 |
| MOLDCAST, 16 x 1 | 816 | 0.567 ± 0.002 | 15.7 ± 1.6 | 12.1 ± 1.8 |
| MOLDCAST, 64 x 1 | 960 | 0.569 ± 0.004 | 16.3 ± 1.4 | 12.7 ± 0.9 |
| MOLDCAST, 2 x 8 | 788 | 0.624 ± 0.003 | 39.6 ± 2.4 | 12.4 ± 0.7 |
| MOLDCAST, 16 x 8 | 928 | 0.621 ± 0.002 | 39.9 ± 1.2 | 12.5 ± 1.4 |
| MOLDCAST, 64 x 8 | 1,408 | 0.621 ± 0.003 | 35.6 ± 1.6 | 13.5 ± 0.3 |
| SparseCL | 1,536 | 0.569 ± 0.002 | 12.1 ± 1.3 | 1.4 ± 0.2 |

## Section 8. What the 16 slots of MOLDCAST 16 x 8 do (per seed; full report: expected_results/interpret_cr16.md)

| Seed | Slots for 80% of a clash | Top slot's share | Edit-type NMI (shuffled) | Attention on edit / chance | Ablation per slot (points) |
| --- | --- | --- | --- | --- | --- |
| 0 | 13.0 | 7% | 0.004 (0.004) | 11% / 11% | 1.8-1.9 |
| 1 | 13.0 | 7% | 0.003 (0.004) | 11% / 11% | 2.0-2.1 |
| 2 | 13.0 | 7% | 0.004 (0.004) | 11% / 11% | 1.9-2.0 |

## Section 8 / Appendix C table. How MOLDCAST 16 x 8 encodes numbers (probe_numbers.py)

| Measure | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| Same number, other template (clash ratio) | 0.06 | 0.05 | 0.05 |
| Word finds its digit (top-1; chance 1%) | 25% | 26% | 27% |
| "six" against "6" (clash ratio) | 0.50 | 0.50 | 0.49 |
| Same last digit, e.g. 6 and 16 (clash ratio) | 0.75 | 0.74 | 0.74 |
| Clash against distance in size (Spearman) | -0.06 | -0.06 | -0.05 |
