#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/liay/linux/miniconda3/envs/notears/bin/python}"
MAX_JOBS="${MAX_JOBS:-4}"

if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer." >&2
    exit 1
fi

if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo "Rolling scheduling requires Bash 5.1 or newer." >&2
    exit 1
fi

# Limit numerical-library threads in each concurrent process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

declare -A pids=()
failed_jobs=0
wait_for_one() {
    local finished_pid status=0
    wait -n -p finished_pid "${!pids[@]}" || status=$?

    # Already-finished children may no longer be eligible for wait -n.
    # Explicit wait still retrieves their saved exit status.
    if [[ -z "${finished_pid:-}" ]]; then
        for finished_pid in "${!pids[@]}"; do
            break
        done
        status=0
        wait "${finished_pid}" || status=$?
    fi

    if (( status != 0 )); then
        failed_jobs=$((failed_jobs + 1))
        echo "Experiment failed (exit ${status}); log: ${pids[$finished_pid]}" >&2
    fi
    unset 'pids[$finished_pid]'
    return 0
}

NODE_COUNTS=(50)
EPSILONS=(0.05)
LAMBDA1_VALUES=(0.1)
# TAU_SETTINGS=(1.0 3.0 10.0 15.0) #"1.0 0.5" "1.0 0.0"
NOISE_TYPES=(gauss)
PRIOR_TYPES=(
    # forbid_edge_pairs
    # forbid_path_pairs
    # forbid_trek_pairs
    # exist_edge_pairs
    exist_path_pairs
    # exist_trek_pairs
    # mix
)
# Configure the prior-rate sweep here; use decimal spelling for output filenames.
PRIOR_RATES=(0.5)
# Each entry is "graph_type:edge_factor". linear_nocurl_joint.py calculates s0 as d times
# edge_factor, giving the ER1, ER2, ER4, and SF4 regimes.
GRAPH_SETTINGS=(
    # "ER:1"
    "ER:2"
    # "ER:4"
    # "SF:4"
)
# Decimal spelling matches the float values in Python output filenames.
# EPSILONS=(0.1 0.01 0.001 0.0001)
cd "${PROJECT_ROOT}"

for seed in {0..9}; do
# for seed in 0; do
    for d in "${NODE_COUNTS[@]}"; do
        # for tau_setting in "${TAU_SETTINGS[@]}"; do
        for graph_setting in "${GRAPH_SETTINGS[@]}"; do
            graph_type="${graph_setting%%:*}"
            edge_factor="${graph_setting##*:}"

            for noise_type in "${NOISE_TYPES[@]}"; do
                for prior_type in "${PRIOR_TYPES[@]}"; do
                    for prior_rate in "${PRIOR_RATES[@]}"; do
                        for epsilon in "${EPSILONS[@]}"; do
                            for lambda1 in "${LAMBDA1_VALUES[@]}"; do
                                echo "Running seed=${seed} d=${d} graph=${graph_type}${edge_factor} noise=${noise_type} prior=${prior_type} rate=${prior_rate} epsilon=${epsilon} lambda1=${lambda1}"

                                output_dir="${PROJECT_ROOT}/linear_nocurl_${prior_type}"
                                log_dir="${output_dir}/log"
                                result_stem="linear_nocurl_init_from_linear_hard_${prior_type}_${graph_type}${edge_factor}_d${d}_${noise_type}_rate${prior_rate}_epsilon${epsilon}_seed${seed}_tau1.0_soft_hard_lambda1${lambda1}_joint"
                                result_file="${output_dir}/${result_stem}.json"
                                log_file="${log_dir}/${result_stem}.log"
                                mkdir -p "${output_dir}" "${log_dir}"

                                (
                                    "${PYTHON_BIN}" -u -m prior_notears.linear_nocurl_joint \
                                        --seed "${seed}" \
                                        --num_nodes "${d}" \
                                        --num_edges_per_node "${edge_factor}" \
                                        --graph_type "${graph_type}" \
                                        --noise "${noise_type}" \
                                        --prior_type "${prior_type}" \
                                        --prior_rate "${prior_rate}" \
                                        --epsilon "${epsilon}" \
                                        --lambda1 "${lambda1}" \
                                        --tau 1.0

                                    if [[ ! -f "${result_file}" ]]; then
                                        echo "Expected result file was not created: ${result_file}" >&2
                                        exit 1
                                    fi
                                ) > "${log_file}" 2>&1 &
                                pids[$!]="${log_file}"
                                if (( ${#pids[@]} >= MAX_JOBS )); then
                                    wait_for_one
                                fi
                            done
                        done
                    done
                done
            done
        done
        # done
    done
done

while (( ${#pids[@]} > 0 )); do
    wait_for_one
done
if (( failed_jobs )); then
    echo "All linear NoCurl experiments finished; ${failed_jobs} experiment(s) failed. Check the logs." >&2
    exit 1
fi
echo "All linear NoCurl experiments completed."

# PYTHON_BIN="$(command -v python)" MAX_JOBS=10 bash prior_notears/run_linear_nocurl.sh
