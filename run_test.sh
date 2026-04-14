#!/usr/bin/env bash
set -e

OUT_DIR=./log_warmup_final_test_new_wd
mkdir -p ${OUT_DIR}

export MASTER_ADDR=127.0.0.1

for K in {1..4}; do
  export MASTER_PORT=$(shuf -i 20000-60000 -n 1)

  echo "===================================="
  echo "Running K=${K}"
  echo "===================================="

  torchrun --nproc_per_node=4 test_main.py \
    --workload=librispeech_conformer \
    --framework=pytorch \
    --submission_path=algorithms/line_search/submission_AdamW_ag_warmup_visual.py \
    --tuning_ruleset=external \
    --tuning_search_space=algorithms/line_search/tuning_search_space_test.json\
    --num_tuning_trials=11 \
    --hparam_start_index=${K} \
    --hparam_end_index=$((K + 1)) \
    --data_dir=/scratch.global/chen8596/mlcommons_data/librispeech \
    --experiment_dir=/scratch.global/chen8596/mlcommons_experiments_TEST_TEST \
    --experiment_name=test_librispeech_large_batch_size \
    --rng_seed=42 \
    --skip_evals 
    # > ${OUT_DIR}/run_K${K+1}.log 2>&1
done
