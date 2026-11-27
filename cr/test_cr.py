"""test_cr.py - unit tests for moldcast_v5 (run from the repo root: python test_cr.py [tiny_encoder])"""
import random
import sys

import torch

import cr_data
from cr_model import SlotEncoder


def test_metrics():
    rng = random.Random(0)
    qrels, results = {}, {}
    for q in range(200):
        qid = f"q{q}"
        rel = rng.sample(range(50), rng.randint(1, 4))
        qrels[qid] = {f"d{d}": 1 for d in rel}
        if q % 7:                                   # some queries without results
            ranked = rng.sample(range(50), 10)
            results[qid] = {f"d{d}": 10 - i + rng.random() * 0.1 for i, d in enumerate(ranked)}
    ref = cr_data.evaluate(qrels, results)
    assert ref["backend"] == "pytrec_eval", "install pytrec_eval to run the parity test"
    import builtins
    real_import = builtins.__import__

    def no_trec(name, *a, **k):
        if name == "pytrec_eval":
            raise ImportError
        return real_import(name, *a, **k)
    builtins.__import__ = no_trec
    try:
        mine = cr_data.evaluate(qrels, results)
    finally:
        builtins.__import__ = real_import
    assert mine["backend"] == "builtin"
    for k in ("NDCG@10", "Recall@10"):
        assert abs(ref[k] - mine[k]) < 1e-9, (k, ref[k], mine[k])
    print(f"ok metrics: builtin NDCG@10/Recall@10 identical to pytrec_eval "
          f"({ref['NDCG@10']:.4f} / {ref['Recall@10']:.4f})")


def test_inner_product(tiny):
    torch.manual_seed(0)
    texts = ["A1 B2 C3 D4", "A1 B2 C3 D1", "E2 F3 A4", "B1 B2 B3 B4 C1", "D2"]
    for n, k, g, act in ((1, 1, 0, "linear"), (4, 3, 0, "linear"), (16, 8, 0, "linear"),
                         (4, 3, 5, "linear"), (16, 8, 16, "linear"), (6, 4, 0, "sphere"),
                         (6, 4, 0, "tanh")):
        m = SlotEncoder(tiny, n_slots=n, value_dim=k, d_model=32, n_heads=2, n_mix=1,
                        gate_dim=g, cos_dim=24 if g else None, value_act=act).eval()
        u = torch.nn.functional.normalize(torch.randn(len(texts), 24), dim=-1)  # stand-in cos emb.
        with torch.no_grad():
            a, v = m(m.tokenize(texts, "cpu"))
            K = m.clash(a, v, a, v)
            assert torch.allclose(K, m.query_vec(a, v) @ m.stored_vec(a, v).T, atol=1e-5)
            assert torch.allclose(K, K.T, atol=1e-5) and K.diagonal().abs().max() < 1e-5
            direct = m.gated_clash(a, v, a, v, u, u)
            sv = m.gate_vectors(m.encode(texts, device="cpu", amp=False, side="stored"), u)
            qv = m.gate_vectors(m.encode(texts, device="cpu", amp=False, side="query"), u)
        assert torch.allclose(direct, qv @ sv.T, atol=1e-5), (direct - qv @ sv.T).abs().max()
        assert sv.size(1) == max(g, 1) * (2 * n + n * k)
        if g:
            assert not torch.allclose(direct, K), "the gate must change the clash"
        if act == "sphere":
            assert torch.allclose(v.norm(dim=-1), torch.ones_like(v.norm(dim=-1)), atol=1e-5)
    print("ok inner product: K and the topic-gated clash both equal <query_vec, stored_vec> exactly "
          "(ungated and gated, several slot sizes, linear / tanh / sphere values); K symmetric; "
          "K(x, x) = 0; sphere values have unit length")


def test_data(data_dir):
    for ds in ("hotpotqa", "arguana"):
        tr = cr_data.load_triples(cr_data.triples_file(data_dir, ds, "train"))
        assert len(tr) > 0 and all(len(t) == 3 for t in tr)
        corpus, queries, qrels = cr_data.load_split(data_dir, ds, "test")
        assert all(isinstance(t, str) for t in corpus.values())
        assert all(q in queries for q in qrels)
        missing = sum(1 for rel in qrels.values() for d in rel if d not in corpus)
        dead = sum(1 for rel in qrels.values() if all(d not in corpus for d in rel))
        assert missing <= 0.01 * sum(len(r) for r in qrels.values()), \
            f"{ds}: {missing} relevant ids missing from the corpus: an id-format problem?"
        if missing:
            print(f"  note: {ds} test has {missing} relevant ids missing from the released corpus "
                  f"({dead} queries unanswerable by any system; same data as SparseCL's evaluation)")
        n_split = sum(1 for p in corpus if p.startswith("test-"))
        assert n_split < len(corpus), "corpus must be the union of all splits"
    print("ok data: triples and retrieval pickles read; corpus = union of splits; relevant ids "
          "found in the corpus")


if __name__ == "__main__":
    tiny = sys.argv[1] if len(sys.argv) > 1 else "/tmp/syn/tiny"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    test_metrics()
    test_inner_product(tiny)
    test_data(data_dir)
    print("all checks passed")
