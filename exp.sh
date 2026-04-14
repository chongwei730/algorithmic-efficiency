#!/bin/bash
#SBATCH --job-name=ls_full
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4 
#SBATCH -p a100-4,apollo_agate
#SBATCH --array=0-4
#SBATCH --output=exp_log/small_%A_%a.out
#SBATCH --error=exp_log/small_%A_%a.err
set -e

NPROC=4
FRAMEWORK=pytorch
K_START=0
K_END=1

BASE_EXPERIMENT_DIR=/scratch.global/chen8596/mlcommons_experiments_train_test
BASE_LOG_DIR=/scratch.global/chen8596/mlcommons_experiments_train_test/logs

######################################
# DATA PATHS
######################################
LIBRISPEECH_DIR=/scratch.global/chen8596/mlcommons_data/librispeech
CIFAR_DIR=/scratch.global/chen8596/mlcommons_data/cifar
WMT_DIR=/scratch.global/chen8596/mlcommons_data/wmt
IMAGENET_DIR=/scratch.global/chen8596/mlcommons_data/imagenet
OGBG_DIR=/scratch.global/chen8596/mlcommons_data/ogbg
FASTMRI_DIR=/scratch.global/chen8596/mlcommons_data/fastmri
LIBRISPEECH_VOCAB=${LIBRISPEECH_DIR}/spm_model.vocab




######################################
# TASKS
######################################
TASKS=(
  wmt
  imagenet_vit
  imagenet_resnet
  # ogbg
  # fastmri
  librispeech_conformer
  librispeech_deepspeech
)

######################################
# METHODS
######################################

# -------- external tuning --------
EXTERNAL_METHODS=(
  # "external_NadamW"
  # "Line_Search_AdamW"
#   "external_amos"
#   "external_baseline"
#   "external_caspr_adaptive"
  # "external_cyclic_lr"
  # "external_generalized_adam"
#   "external_lawa_ema"
#   "external_lawa_queue"
  # "external_nadamp"
  # "external_schedule_free_adamw"
  # "external_schedule_free_prodigy"
  # "external_shampoo_submission"
)

# -------- self tuning --------
SELF_METHODS=(
  # "self_NadamW"
#   "self_adamg"
#   "self_baseline"
#   "self_nadamw_sequential"
  "self_schedule_free_adamw"
#   "self_sinv6"
#   "self_sinv6_75"
# "self_Line_Search_AdamW"
)


######################################
# METHOD CONFIG (maps name → paths)
######################################

get_method_config () {
  local method=$1

  case ${method} in
    external_NadamW)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/baselines/external_tuning/pytorch_nadamw_full_budget.py
      TUNING_SEARCH_SPACE=algorithms/baselines/external_tuning/tuning_search_space.json
      ;;
    Line_Search_AdamW)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/line_search/submission_AdamW_ag_warmup.py
      TUNING_SEARCH_SPACE=algorithms/line_search/tuning_search_space_AdamW_short_interval_loose_c.json
      ;;
    external_amos)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/amos/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/amos/tuning_search_space.json
      ;;
    external_baseline)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/baseline/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/baseline/tuning_search_space.json
      ;;
    external_caspr_adaptive)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/caspr_adaptive/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/caspr_adaptive/tuning_search_space.json
      ;;
    external_cyclic_lr)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/cyclic_lr/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/cyclic_lr/tuning_search_space.json
      ;;
    external_generalized_adam)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/generalized_adam/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/generalized_adam/tuning_search_space.json
      ;;
    external_lawa_ema)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/lawa_ema/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/lawa_ema/tuning_search_space.json
      ;;
    external_lawa_queue)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/lawa_queue/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/lawa_queue/tuning_search_space.json
      ;;
    external_nadamp)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/nadamp/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/nadamp/tuning_search_space.json
      ;;
    external_schedule_free_adamw)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/schedule_free_adamw/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/schedule_free_adamw/tuning_search_space.json
      ;;
    external_schedule_free_prodigy)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/schedule_free_prodigy/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/schedule_free_prodigy/tuning_search_space.json
      ;;
    external_shampoo_submission)
      TUNING_RULESET=external
      SUBMISSION_PATH=algorithms/submissions/external_tuning/shampoo_submission/submission.py
      TUNING_SEARCH_SPACE=algorithms/submissions/external_tuning/shampoo_submission/tuning_search_space.json
      ;;
    self_NadamW)
      TUNING_RULESET=self
      SUBMISSION_PATH=algorithms/baselines/self_tuning/submission.py
      TUNING_SEARCH_SPACE=algorithms/baselines/self_tuning/tuning_search_space.json
      ;;
    self_adamg)
      TUNING_RULESET=self
      SUBMISSION_PATH=algorithms/submissions/self_tuning/adamg/submission.py
      ;;
    self_baseline)
      TUNING_RULESET=self
      SUBMISSION_PATH=algorithms/submissions/self_tuning/baseline/submission.py
      ;;
    self_nadamw_sequential)
      TUNING_RULESET=self
      SUBMISSION_PATH=algorithms/submissions/self_tuning/nadamw_sequential/submission.py
      ;;
    self_schedule_free_adamw)
      TUNING_RULESET=self
      SUBMISSION_PATH=algorithms/submissions/self_tuning/schedule_free_adamw/submission.py
      ;;
    self_sinv6)
      TUNING_RULESET=self
      SUBMISSION_PATH=algorithms/submissions/self_tuning/sinv6/submission.py
      ;;
    self_sinv6_75)
      TUNING_RULESET=self
      SUBMISSION_PATH=algorithms/submissions/self_tuning/sinv6_75/submission.py
      ;;
    self_Line_Search_AdamW)
      TUNING_RULESET=self
      SUBMISSION_PATH=algorithms/line_search/submission_AdamW_self_tuning.py
      ;;
    *)
      echo "Unknown method: ${method}"
      exit 1
      ;;
  esac
}



######################################
extra_args_for_task () {
  local task=$1
  case ${task} in
    cifar)
      echo "--data_dir=${CIFAR_DIR}"
      ;;
    librispeech_conformer|librispeech_deepspeech)
      echo "--data_dir=${LIBRISPEECH_DIR} \
            --librispeech_tokenizer_vocab_path=${LIBRISPEECH_VOCAB}"
      ;;
    wmt)
      echo "--data_dir=${WMT_DIR}"
      ;;
    imagenet_vit|imagenet_resnet)
      echo "--data_dir=${IMAGENET_DIR} \
            --imagenet_v2_data_dir=${IMAGENET_DIR}"
      ;;
    ogbg)
      echo "--data_dir=${OGBG_DIR}"
      ;;
    fastmri)
      echo "--data_dir=${FASTMRI_DIR}"
      ;;
    *)
      echo ""
      ;;
  esac
}

######################################
# MAIN LOOP (Scheme B)
# Map a single SLURM array index (or script arg) to one job: (ruleset, method, task, K)
######################################

# Build flattened job list: entries are "<ruleset>|<method>|<task>|<K>"
JOBS=()

# External tuning jobs (each K is an hparam index)
for method in "${EXTERNAL_METHODS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for ((K=${K_START}; K<${K_END}; K++)); do
      JOBS+=("external|${method}|${TASK}|${K}")
    done
  done
done

# Self tuning jobs (each K is a run index)
for method in "${SELF_METHODS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for ((K=${K_START}; K<${K_END}; K++)); do
      JOBS+=("self|${method}|${TASK}|${K}")
    done
  done
done

TOTAL_JOBS=${#JOBS[@]}

if [ ${TOTAL_JOBS} -eq 0 ]; then
  echo "No jobs to run (check EXTERNAL_METHODS/SELF_METHODS/TASKS/K_START/K_END)"
  exit 1
fi

# Helpers: allow printing count or listing jobs without running
if [ "$1" = "--count" ]; then
  echo "${TOTAL_JOBS}"
  exit 0
fi

if [ "$1" = "--list" ]; then
  for i in "${!JOBS[@]}"; do
    echo "${i} ${JOBS[$i]}"
  done
  exit 0
fi

# Determine which job index to run
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
  IDX=${SLURM_ARRAY_TASK_ID}
elif [ -n "$1" ]; then
  IDX=$1
else
  echo "SLURM_ARRAY_TASK_ID not set and no job index provided."
  echo "Submit with: sbatch --array=0-$((TOTAL_JOBS-1))%<concurrency> exp.sh"
  exit 1
fi

if [ ${IDX} -lt 0 ] || [ ${IDX} -ge ${TOTAL_JOBS} ]; then
  echo "Job index ${IDX} out of range (0..$((TOTAL_JOBS-1)))"
  exit 1
fi

JOB=${JOBS[${IDX}]}
IFS='|' read -r JOB_TYPE METHOD TASK K <<< "${JOB}"

get_method_config "${METHOD}"

if [ "${JOB_TYPE}" = "external" ]; then
  EXP_NAME="${METHOD}"
  EXP_DIR="${BASE_EXPERIMENT_DIR}/"
  LOG_DIR="${BASE_LOG_DIR}/${TASK}/${METHOD}"
  LOG_FILE="${LOG_DIR}/hparam_${K}.log"

  mkdir -p "${EXP_DIR}" "${LOG_DIR}"

  echo "=============================================="
  echo "Method=${METHOD} (external)"
  echo "Task=${TASK}, hparam_index=${K}, job_idx=${IDX}/${TOTAL_JOBS}"
  echo "=============================================="


  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=$(shuf -i 20000-60000 -n 1)
  torchrun --nproc_per_node=${NPROC} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    submission_runner.py \
    --framework=${FRAMEWORK} \
    --submission_path=${SUBMISSION_PATH} \
    --tuning_ruleset=${TUNING_RULESET} \
    --tuning_search_space=${TUNING_SEARCH_SPACE} \
    --num_tuning_trials=5 \
    --rng_seed=42 \
    --hparam_start_index=${K} \
    --hparam_end_index=$((K + 1)) \
    --experiment_dir="${EXP_DIR}" \
    --experiment_name="${EXP_NAME}" \
    --workload="${TASK}" \
    $(extra_args_for_task "${TASK}") \
    > "${LOG_FILE}" 2>&1

elif [ "${JOB_TYPE}" = "self" ]; then
  EXP_NAME="${METHOD}"
  EXP_DIR="${BASE_EXPERIMENT_DIR}/"
  LOG_DIR="${BASE_LOG_DIR}/${TASK}/${METHOD}"
  LOG_FILE="${LOG_DIR}/run_${K+1}.log"

  mkdir -p "${EXP_DIR}" "${LOG_DIR}"

  echo "=============================================="
  echo "Method=${METHOD} (self)"
  echo "Task=${TASK}, run=${K}, job_idx=${IDX}/${TOTAL_JOBS}"
  echo "=============================================="

  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
  torchrun --nproc_per_node=${NPROC} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    submission_runner.py \
    --framework=${FRAMEWORK} \
    --submission_path=${SUBMISSION_PATH} \
    --tuning_ruleset=${TUNING_RULESET} \
    --experiment_dir="${EXP_DIR}" \
    --experiment_name="${EXP_NAME}" \
    --workload="${TASK}" \
    $(extra_args_for_task "${TASK}") \
    > "${LOG_FILE}" 2>&1
else
  echo "Unknown job type: ${JOB_TYPE}"
  exit 1
fi
