#!/usr/bin/env bash
# setup.sh - run ONCE, from this folder, on a machine with internet access (e.g. a login node).
#
#   1. SparseCL's own environment (sparsecl_env/): its code needs transformers 4.36 and older
#      sentence-transformers / beir / datasets, so it lives apart from the main environment
#   2. SparseCL's repository (sparsecl_repo/), checked out at the commit used for the paper, with
#      test_qf.py = its test script plus a query-free switch (three added lines), and its data
#      folder linked to ./data
#   3. downloads into the Hugging Face cache, so that jobs can run on offline compute nodes:
#      bge-base, the two OPUS-MT models, VitaminC, SparseCL's released bge checkpoints, and
#      SparseCL's benchmark data (ArguAna, MSMARCO, HotpotQA) into ./data
#
# The main environment (the one you run run_all.sh from) is described in README.md.
set -euo pipefail
cd "$(dirname "$0")"
ENV=sparsecl_env
REPO=sparsecl_repo
SPARSECL_COMMIT=14fcd8bcfeef2603f37b1e805cd3626800430711     # the commit used for the paper
mkdir -p data

command -v uv >/dev/null || { echo "uv not found (pip install uv)"; exit 1; }
[ -d $ENV ] || uv venv $ENV --python 3.11
VIRTUAL_ENV=$PWD/$ENV uv pip install "torch==2.1.2" "transformers==4.36.2" "tokenizers==0.15.0" \
  "sentence-transformers==2.2.2" "beir==2.0.0" "faiss-cpu==1.7.4" "numpy==1.25.2" \
  "huggingface-hub==0.20.2" "accelerate==0.21.0" "datasets==2.14.4" "pyarrow==14.0.2" \
  "fsspec==2023.6.0" pandas tqdm scipy scikit-learn
$ENV/bin/python -c "import numpy, pyarrow, datasets, beir.retrieval.models, faiss, transformers; \
print('SparseCL environment ok: numpy', numpy.__version__, 'transformers', transformers.__version__)"

[ -d $REPO ] || git clone https://github.com/xuhaike/SparseCL $REPO
git -C $REPO checkout -q $SPARSECL_COMMIT
echo "SparseCL at commit $(git -C $REPO rev-parse HEAD)"
[ -e $REPO/data ] || ln -s ../data $REPO/data
$ENV/bin/python - <<'PY'
src = open("sparsecl_repo/test_contradiction_faiss_final.py").read()
anchor = "    query_embeddings=query_embeddings.to(torch.float32).numpy()\n"
assert src.count(anchor) == 1, "SparseCL's test script changed: the query-free patch anchor is not unique"
patch = anchor + ('    if os.environ.get("SPARSECL_QUERY_FREE"):   # added: query-free control\n'
                  '        query_embeddings[:] = query_embeddings.mean(axis=0, keepdims=True)\n'
                  '        print("query-free: every query uses the average query sparse embedding")\n')
open("sparsecl_repo/test_qf.py", "w").write(src.replace(anchor, patch))
print("wrote sparsecl_repo/test_qf.py (SparseCL's test script + query-free switch)")
PY

# downloads (cached in HF_HOME, shared by both environments)
python download.py --sparsecl_data
# SparseCL's environment loads bge through sentence-transformers 2.2.2: cache it there too
$ENV/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5'); print('cached bge for beir')"

echo "setup done: see README.md for the experiments (bash run_all.sh round1, round2, ...)"
