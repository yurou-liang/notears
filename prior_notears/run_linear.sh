#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/liay/linux/miniconda3/envs/notears/bin/python}"
PRIOR_RATE="${PRIOR_RATE:-0.5}"
MAX_JOBS="${MAX_JOBS:-4}"

if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer." >&2
    exit 1
fi

# Limit numerical-library threads in each concurrent process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

pids=()
wait_for_batch() {
    local pid
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    pids=()
    if (( failed )); then
        echo "One or more experiments failed; check the logs." >&2
        return 1
    fi
}

NODE_COUNTS=(100)
EPSILONS=(0.1)
NOISE_TYPES=(gauss)
LOSS_TYPE=(both)
PRIOR_TYPES=(
    # forbid_edge_pairs
    # forbid_path_pairs
    # forbid_trek_pairs
    exist_edge_pairs
    # exist_path_pairs
    # exist_trek_pairs
)

# Each entry is "graph_type:edge_factor". linear.py calculates s0 as d times
# edge_factor, giving the ER1, ER2, ER4, and SF4 regimes.
GRAPH_SETTINGS=(
    "ER:1"
    # "ER:2"
    # "ER:4"
    # "SF:4"
)
# Decimal spelling matches the float values in Python output filenames.
# EPSILONS=(0.1 0.01 0.001 0.0001)
cd "${PROJECT_ROOT}"

for seed in {0..9}; do
    for d in "${NODE_COUNTS[@]}"; do
        for graph_setting in "${GRAPH_SETTINGS[@]}"; do
            graph_type="${graph_setting%%:*}"
            edge_factor="${graph_setting##*:}"

            for noise_type in "${NOISE_TYPES[@]}"; do
                for prior_type in "${PRIOR_TYPES[@]}"; do
                    for epsilon in "${EPSILONS[@]}"; do
                        echo "Running seed=${seed} d=${d} graph=${graph_type}${edge_factor} noise=${noise_type} prior=${prior_type} rate=${PRIOR_RATE} epsilon=${epsilon}"

                        output_dir="${PROJECT_ROOT}/linear_${prior_type}"
                        log_dir="${output_dir}/log"
                        result_stem="linear_${prior_type}_${graph_type}${edge_factor}_d${d}_${noise_type}_rate${PRIOR_RATE}_epsilon${epsilon}_seed${seed}"
                        result_file="${output_dir}/${result_stem}.json"
                        log_file="${log_dir}/${result_stem}.log"
                        mkdir -p "${output_dir}" "${log_dir}"

                        (
                            "${PYTHON_BIN}" -m prior_notears.linear \
                                --seed "${seed}" \
                                --num_nodes "${d}" \
                                --num_edges_per_node "${edge_factor}" \
                                --graph_type "${graph_type}" \
                                --loss_type "${LOSS_TYPE}" \
                                --noise "${noise_type}" \
                                --prior_type "${prior_type}" \
                                --prior_rate "${PRIOR_RATE}" \
                                --epsilon "${epsilon}" \
                                > "${log_file}" 2>&1

                            if [[ ! -f "${result_file}" ]]; then
                                echo "Expected result file was not created: ${result_file}" >&2
                                exit 1
                            fi
                        ) &
                        pids+=("$!")
                        if (( ${#pids[@]} >= MAX_JOBS )); then
                            wait_for_batch
                        fi
                    done
                done
            done
        done
    done
done

wait_for_batch
echo "All linear experiments completed."
