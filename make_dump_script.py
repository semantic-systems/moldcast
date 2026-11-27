"""make_dump_script.py - write sparsecl_repo/dump_emb.py: SparseCL's own test script, stopped right
after it has computed its four embedding sets (cosine and sparse, for passages and queries), which
it saves to the file named by SPARSECL_DUMP. Everything before that point is their code, unchanged,
so the embeddings are exactly the ones their evaluation scores."""
src = open("sparsecl_repo/test_contradiction_faiss_final.py").read()
anchor = "    sqrt_dim=math.sqrt(input_embeddings.shape[1])\n"
assert src.count(anchor) == 1, "their test script changed: the dump anchor is not unique"
dump = anchor + (
    '    if os.environ.get("SPARSECL_DUMP"):   # added: save the embeddings and stop\n'
    '        import numpy as _np, sys as _sys\n'
    '        _np.savez(os.environ["SPARSECL_DUMP"], cos_d=cos_input_embeddings, cos_q=cos_query_embeddings,\n'
    '                  sp_d=input_embeddings, sp_q=query_embeddings, pids=_np.array(passage_id),\n'
    '                  qids=_np.array(list(queries.keys())))\n'
    '        print("saved embeddings to", os.environ["SPARSECL_DUMP"])\n'
    '        _sys.exit(0)\n')
open("sparsecl_repo/dump_emb.py", "w").write(src.replace(anchor, dump))
print("wrote sparsecl_repo/dump_emb.py")
