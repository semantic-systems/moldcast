#!/usr/bin/env bash
# run_all.sh - every experiment in the paper, as SLURM jobs (or, with LOCAL=1, one after the other
# on this machine). Run  bash setup.sh  once first. Finished steps are skipped, so a mode can be
# rerun safely after a failure. Each mode ends with  results/paper_tables.md  (make_table.py).
#
#   bash run_all.sh round1     Section 5 (Tables 1 and 2, Appendix B): zero-shot cosine, MOLDCAST
#                              16x8 and 2x1 on SparseCL's benchmarks (3 seeds), query-free, the style
#                              probe and changed-fact test (cr_analysis.py); SparseCL's released
#                              models normally and query-free; slot readings (slot_inspect.py)
#   bash run_all.sh round2     Sections 6-7 (Table 3, Appendix D): builds the symmetric VitaminC
#                              benchmark and checks it; zero-shot cosine; MOLDCAST in six
#                              configurations (3 seeds); SparseCL trained on the benchmark (3 seeds)
#   bash run_all.sh diagnose   (after round2) pairwise accuracy and the test-only corpus (Table 3)
#   bash run_all.sh validate   (after diagnose) our training of SparseCL on its own HotpotQA data
#                              against its released model, and the Hoyer check (Appendix C)
#   bash run_all.sh analysis   (after round2) Section 8: slot analysis, value clusters and the
#                              number probe of MOLDCAST 16x8
#   bash run_all.sh tables     only rebuild results/paper_tables.md
#
# Settings (environment variables): CPUS, GPU (sbatch GPU option), TIME_TRAIN, TIME_EVAL,
# LOCAL=1 (no SLURM). Logs: logs/slurm/ (SLURM) or logs/local/ (LOCAL=1); job scripts: logs/jobs/.
set -euo pipefail
MODE=${1:-}
case "$MODE" in round1|round2|diagnose|validate|analysis|tables) ;;
  *) echo "usage: bash run_all.sh round1|round2|diagnose|validate|analysis|tables"; exit 1 ;; esac
cd "$(dirname "$0")"
SEEDS="0 1 2"
BGE=BAAI/bge-base-en-v1.5
CPUS=${CPUS:-8}
GPU=${GPU:---gpus-per-node=1}
TIME_TRAIN=${TIME_TRAIN:-24:00:00}
TIME_EVAL=${TIME_EVAL:-12:00:00}
# MOLDCAST configurations: slots x numbers per value (stored per passage: 768 + slots x (numbers + 2))
declare -A ARM_FLAGS=([cr16]="--n_slots 16 --value_dim 8" [cr2]="--n_slots 2 --value_dim 1" \
  [cr2x8]="--n_slots 2 --value_dim 8" [cr16x1]="--n_slots 16 --value_dim 1" \
  [cr64x1]="--n_slots 64 --value_dim 1" [cr64x8]="--n_slots 64 --value_dim 8")
ROUND1_ARMS="cr16 cr2"                           # Table 1
ROUND2_ARMS="cr16 cr2 cr2x8 cr16x1 cr64x1 cr64x8"  # Table 3 (cr16, cr2) and Appendix D (all six)
mkdir -p logs/slurm logs/local logs/jobs results

if [ "$MODE" != tables ]; then
  [ -x sparsecl_env/bin/python ] && [ -f sparsecl_repo/test_qf.py ] || {
    echo "run  bash setup.sh  first (on a machine with internet access)"; exit 1; }
fi

SB="sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task=$CPUS --kill-on-invalid-dep=yes"
ALL=""
submit() {   # name time dependency-ids command -> job id; the command is written to logs/jobs/
  local name=$1 time=$2 deps=$3; shift 3
  local f=logs/jobs/$name.sh
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n%s\n' "$PWD" "$*" > "$f"
  if [ "${LOCAL:-0}" = 1 ]; then          # run it here, now, in submission (= dependency) order
    echo "  running $name (log: logs/local/$name.out)" >&2
    if ! bash "$f" > "logs/local/$name.out" 2>&1; then
      echo "  $name FAILED: see logs/local/$name.out; fix it and rerun (finished steps are skipped)" >&2
      kill -TERM $$
      return 1
    fi
    echo "local-$name"
    return 0
  fi
  local dep=""
  [ -n "$deps" ] && dep="--dependency=afterok:$(echo $deps | tr ' ' ':')"
  $SB $GPU --time=$time --job-name=$name --output=logs/slurm/%x-%j.out $dep "$f"
}

scl_test() {   # dataset model seq_len write_path [queryfree] -> SparseCL's test script, dev then test
  local ds=$1 model=$2 len=$3 wp=$4 qf=${5:-}
  local env=""; [ -n "$qf" ] && env="SPARSECL_QUERY_FREE=1"
  echo "for split in dev test; do $env ../sparsecl_env/bin/python test_qf.py --dataset_name $ds \
    --split \$split --model_name_or_path $model --cos_model_name bge \
    --write_path test_results/\${split}_$wp --pooler_type avg --metric both_sum --max_seq_length $len; done"
}

moldcast_job() {   # configuration dataset seed dependency-ids [extra command] [done file]
  # trains MOLDCAST, chooses alpha on dev, evaluates on test, normally and query-free
  local arm=$1 ds=$2 s=$3 deps=$4 extra=${5:-} done_file=${6:-}
  local name=${arm}_${ds}_s$s
  local R=runs/$name/last
  [ -n "$done_file" ] || done_file=cr/$R/eval_${ds}_test_queryfree.json
  [ -f "$done_file" ] && return 0
  submit $name $TIME_TRAIN "$deps" "cd cr \
    && { [ -f $R/model.pt ] || python cr_train.py --dataset $ds --data_dir ../data \
         --out runs/$name --seed $s --gate_dim 0 ${ARM_FLAGS[$arm]} --backbone $BGE \
         --cos_model $BGE --grad_ckpt --no_wandb; } \
    && python cr_eval.py --dataset $ds --data_dir ../data --split dev --model $R \
    && python cr_eval.py --dataset $ds --data_dir ../data --split test --model $R \
    && python cr_eval.py --dataset $ds --data_dir ../data --split dev --model $R --query_free \
    && python cr_eval.py --dataset $ds --data_dir ../data --split test --model $R --query_free$extra"
}

cos_job() {   # dataset dependency-ids: zero-shot cosine on the test split
  local ds=$1 deps=$2
  [ -f results/cos_${ds}_test.json ] && return 0
  submit cos_$ds $TIME_EVAL "$deps" "cd cr && python cr_eval.py --dataset $ds --data_dir ../data \
    --split test --cos_model $BGE --out ../results/cos_${ds}_test.json"
}

# ---------------------------------------------------------------- round 1 (Section 5)
if [ "$MODE" = round1 ]; then
  for f in data/arguana_test_retrieval_final.pkl data/msmarco_test_retrieval_gpt4_final.pkl \
           data/hotpotqa_test_retrieval_gpt4_final.pkl; do
    [ -f $f ] || { echo "missing $f: SparseCL's data is downloaded by setup.sh"; exit 1; }
  done
  declare -A JOB
  for ds in arguana msmarco hotpotqa; do
    id=$(cos_job $ds ""); [ -n "$id" ] && { echo "cos_$ds: job $id"; ALL="$ALL $id"; }
    for arm in $ROUND1_ARMS; do
      for s in $SEEDS; do
        name=${arm}_${ds}_s$s
        extra=""; done_file=""
        if [ $arm = cr16 ]; then   # the style probe and the changed-fact test (Section 5)
          extra=" && python cr_analysis.py --dataset $ds --data_dir ../data --model runs/$name/last \
            --out ../results/analysis_$name.json"
          done_file=results/analysis_$name.json
        fi
        id=$(moldcast_job $arm $ds $s "" "$extra" "$done_file")
        [ -n "$id" ] && { echo "$name: job $id"; ALL="$ALL $id"; JOB[$name]=$id; }
      done
    done
  done
  # SparseCL's released bge models on their own benchmarks, normally and query-free (Tables 1, 2)
  if [ ! -f sparsecl_repo/test_results/test_hotpotqa_bge_zeroshot_hoyer_qf/hotpotqa_parallel_output ]; then
    CMD="cd sparsecl_repo"
    for ds in arguana msmarco hotpotqa; do
      len=256; [ $ds = arguana ] && len=512
      CMD="$CMD && $(scl_test $ds SparseCL/BGE-SparseCL-$ds $len ${ds}_bge_zeroshot_hoyer) \
        && $(scl_test $ds SparseCL/BGE-SparseCL-$ds $len ${ds}_bge_zeroshot_hoyer_qf qf)"
    done
    id=$(submit sparsecl_released $TIME_TRAIN "" "$CMD")
    echo "sparsecl_released: job $id"; ALL="$ALL $id"
  fi
  # slot readings (Appendix B): extremes of individual slots, HotpotQA, seed 0
  if [ ! -f cr/runs/cr2_hotpotqa_s0/last/slot_inspect_hotpotqa.md ]; then
    deps="${JOB[cr16_hotpotqa_s0]:-} ${JOB[cr2_hotpotqa_s0]:-}"
    CMD="cd cr && { [ -f ../data/probe_hotpotqa.jsonl ] || python probe_build.py --dataset hotpotqa \
      --data_dir ../data --out ../data/probe_hotpotqa.jsonl; }"
    for r in cr16_hotpotqa_s0 cr2_hotpotqa_s0; do
      CMD="$CMD && python slot_inspect.py --dataset hotpotqa --data_dir ../data --model runs/$r/last \
        --probe ../data/probe_hotpotqa.jsonl"
    done
    id=$(submit slot_inspect $TIME_EVAL "$deps" "$CMD")
    echo "slot_inspect: job $id"; ALL="$ALL $id"
  fi
fi

# ---------------------------------------------------------------- round 2 (Sections 6-7)
if [ "$MODE" = round2 ]; then
  BUILD=""
  if [ ! -f data/vitaminc_test_retrieval_gpt4_final.pkl ]; then
    BUILD=$(submit build_vitaminc $TIME_EVAL "" "python build_vitaminc.py --out data \
      && python check_benchmark.py --data data")
    echo "build_vitaminc: job $BUILD"; ALL="$ALL $BUILD"
  fi
  id=$(cos_job vitaminc "$BUILD"); [ -n "$id" ] && { echo "cos_vitaminc: job $id"; ALL="$ALL $id"; }
  for arm in $ROUND2_ARMS; do
    for s in $SEEDS; do
      id=$(moldcast_job $arm vitaminc $s "$BUILD")
      [ -n "$id" ] && { echo "${arm}_vitaminc_s$s: job $id"; ALL="$ALL $id"; }
    done
  done
  # SparseCL trained on the benchmark with its own code and settings, normally and query-free
  for s in $SEEDS; do
    M=results/vitaminc-bge-sparsity-s$s
    [ -f sparsecl_repo/test_results/test_vitaminc_bge_sparsity_s${s}_qf/vitaminc_parallel_output ] && continue
    id=$(submit sparsecl_vitaminc_s$s $TIME_TRAIN "$BUILD" "cd sparsecl_repo \
      && { [ -f $M/config.json ] || ../sparsecl_env/bin/python train.py --model_name our_bge \
           --model_name_or_path $BGE --train_file data/vitaminc_train_gpt4_final.csv \
           --eval_file data/vitaminc_dev_gpt4_final.csv --output_dir $M --gradient_checkpointing True \
           --num_train_epochs 3 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
           --gradient_accumulation_steps 1 --learning_rate 2e-5 --max_seq_length 256 \
           --pad_to_max_length True --pooler_type avg --overwrite_output_dir --temp 0.02 \
           --loss_type sparsity --dataloader_drop_last True --do_train --do_eval --fp16 --seed $s; } \
      && $(scl_test vitaminc $M 256 vitaminc_bge_sparsity_s$s) \
      && $(scl_test vitaminc $M 256 vitaminc_bge_sparsity_s${s}_qf qf)")
    echo "sparsecl_vitaminc_s$s: job $id"; ALL="$ALL $id"
  done
fi

# ---------------------------------------------------------------- diagnose (Table 3)
if [ "$MODE" = diagnose ]; then
  for s in $SEEDS; do
    [ -f sparsecl_repo/results/vitaminc-bge-sparsity-s$s/config.json ] || {
      echo "run  bash run_all.sh round2  first (SparseCL seed $s is not trained)"; exit 1; }
  done
  if [ ! -f results/diagnose_ours.json ]; then
    id=$(submit diag_moldcast $TIME_EVAL "" "python diagnose_ours.py --data data --runs cr/runs \
      --cos_model $BGE --arms $ROUND2_ARMS --out results/diagnose_ours.json")
    echo "diag_moldcast: job $id"; ALL="$ALL $id"
  fi
  if [ ! -f results/diagnose_sparsecl.json ]; then
    DEPS=""
    for s in $SEEDS; do
      [ -f results/sparsecl_emb_s$s.npz ] && continue
      id=$(submit diag_dump_s$s $TIME_EVAL "" "python make_dump_script.py && cd sparsecl_repo \
        && SPARSECL_DUMP=../results/sparsecl_emb_s$s.npz ../sparsecl_env/bin/python dump_emb.py \
           --dataset_name vitaminc --split test --model_name_or_path results/vitaminc-bge-sparsity-s$s \
           --cos_model_name bge --write_path test_results/diag_dump_s$s --pooler_type avg \
           --metric both_sum --max_seq_length 256")
      echo "diag_dump_s$s: job $id"; ALL="$ALL $id"; DEPS="$DEPS $id"
    done
    id=$(submit diag_sparsecl $TIME_EVAL "$DEPS" "python diagnose_sparsecl.py --data data \
      --out results/diagnose_sparsecl.json")
    echo "diag_sparsecl: job $id"; ALL="$ALL $id"
  fi
fi

# ---------------------------------------------------------------- validate (Appendix C)
if [ "$MODE" = validate ]; then
  [ -f data/hotpotqa_train_gpt4_final.csv ] || { echo "SparseCL's HotpotQA data is missing: bash setup.sh"; exit 1; }
  for s in $SEEDS; do
    [ -f results/sparsecl_emb_s$s.npz ] || { echo "run  bash run_all.sh diagnose  first"; exit 1; }
  done
  M=results/hotpotqa-bge-sparsity-val
  DUMP="python make_dump_script.py && cd sparsecl_repo"
  VDEPS=""
  if [ ! -f results/val_emb_ours.npz ]; then
    id=$(submit val_ours $TIME_TRAIN "" "cd sparsecl_repo \
      && { [ -f $M/config.json ] || ../sparsecl_env/bin/python train.py --model_name our_bge \
           --model_name_or_path $BGE --train_file data/hotpotqa_train_gpt4_final.csv \
           --eval_file data/hotpotqa_dev_gpt4_final.csv --output_dir $M --gradient_checkpointing True \
           --num_train_epochs 3 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
           --gradient_accumulation_steps 1 --learning_rate 2e-5 --max_seq_length 256 \
           --pad_to_max_length True --pooler_type avg --overwrite_output_dir --temp 0.02 \
           --loss_type sparsity --dataloader_drop_last True --do_train --do_eval --fp16 --seed 0; } \
      && $(scl_test hotpotqa $M 256 hotpotqa_bge_sparsity_val) \
      && cd .. && $DUMP && SPARSECL_DUMP=../results/val_emb_ours.npz ../sparsecl_env/bin/python dump_emb.py \
         --dataset_name hotpotqa --split test --model_name_or_path $M --cos_model_name bge \
         --write_path test_results/val_dump_ours --pooler_type avg --metric both_sum --max_seq_length 256")
    echo "val_ours: job $id"; ALL="$ALL $id"; VDEPS="$VDEPS $id"
  fi
  if [ ! -f results/val_emb_released.npz ]; then
    id=$(submit val_released $TIME_EVAL "" "$DUMP && SPARSECL_DUMP=../results/val_emb_released.npz \
      ../sparsecl_env/bin/python dump_emb.py --dataset_name hotpotqa --split test \
      --model_name_or_path SparseCL/BGE-SparseCL-hotpotqa --cos_model_name bge \
      --write_path test_results/val_dump_released --pooler_type avg --metric both_sum --max_seq_length 256")
    echo "val_released: job $id"; ALL="$ALL $id"; VDEPS="$VDEPS $id"
  fi
  if [ ! -f results/validate_vitaminc.json ]; then
    id=$(submit val_check $TIME_EVAL "$VDEPS" "python check_sparsecl_training.py --dataset hotpotqa \
      --dumps results/val_emb_released.npz results/val_emb_ours.npz --out results/validate_hotpotqa.json \
      && python check_sparsecl_training.py --out results/validate_vitaminc.json")
    echo "val_check: job $id"; ALL="$ALL $id"
  fi
fi

# ---------------------------------------------------------------- analysis (Section 8)
if [ "$MODE" = analysis ]; then
  for s in $SEEDS; do
    [ -f cr/runs/cr16_vitaminc_s$s/last/alpha_vitaminc_full.json ] || {
      echo "MOLDCAST 16x8 seed $s is not trained and evaluated yet: bash run_all.sh round2"; exit 1; }
  done
  if [ ! -f results/interpret_cr16.json ]; then
    id=$(submit slot_analysis $TIME_EVAL "" "python interpret_slots.py --data data --runs cr/runs \
      --arm cr16 --cos_model $BGE --out results/interpret_cr16.json")
    echo "slot_analysis: job $id"; ALL="$ALL $id"
  fi
  if [ ! -f results/explore_cr16.json ]; then
    id=$(submit value_clusters $TIME_EVAL "" "python explore_slots.py --data data --runs cr/runs \
      --arm cr16 --out results/explore_cr16.json")
    echo "value_clusters: job $id"; ALL="$ALL $id"
  fi
  if [ ! -f results/probe_numbers_cr16.json ]; then
    id=$(submit number_probe $TIME_EVAL "" "python probe_numbers.py --runs cr/runs --arm cr16 \
      --out results/probe_numbers_cr16.json")
    echo "number_probe: job $id"; ALL="$ALL $id"
  fi
fi

# ---------------------------------------------------------------- tables
if [ -n "${ALL// /}" ] && [ "${LOCAL:-0}" != 1 ]; then
  printf '#!/usr/bin/env bash\ncd %q\npython make_table.py > results/paper_tables.md\n' "$PWD" > logs/jobs/tables.sh
  id=$(sbatch --parsable --time=01:00:00 --job-name=tables --output=logs/slurm/%x-%j.out \
       --dependency=afterany:$(echo $ALL | tr ' ' ':') logs/jobs/tables.sh)
  echo "tables: job $id (results/paper_tables.md when everything has finished)"
else
  python make_table.py > results/paper_tables.md && echo "results/paper_tables.md written"
fi
