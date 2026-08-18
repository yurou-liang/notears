#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/notears/bin/python}"
PRIOR_RATE="${PRIOR_RATE:-0.5}"

NODE_COUNTS=(5)
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

cd "${PROJECT_ROOT}"

for seed in {0..9}; do
    for d in "${NODE_COUNTS[@]}"; do
        for graph_setting in "${GRAPH_SETTINGS[@]}"; do
            graph_type="${graph_setting%%:*}"
            edge_factor="${graph_setting##*:}"

            for noise_type in "${NOISE_TYPES[@]}"; do
                for prior_type in "${PRIOR_TYPES[@]}"; do
                    echo "Running seed=${seed} d=${d} graph=${graph_type}${edge_factor} noise=${noise_type} prior=${prior_type} rate=${PRIOR_RATE}"

                    output_dir="${PROJECT_ROOT}/linear_${prior_type}"
                    log_dir="${output_dir}/log"
                    result_stem="linear_${prior_type}_${graph_type}${edge_factor}_d${d}_${noise_type}_rate${PRIOR_RATE}_seed${seed}"
                    result_file="${output_dir}/${result_stem}.json"
                    log_file="${log_dir}/${result_stem}.log"
                    mkdir -p "${output_dir}" "${log_dir}"

                    "${PYTHON_BIN}" -m prior_notears.linear \
                        --seed "${seed}" \
                        --num_nodes "${d}" \
                        --num_edges "${edge_factor}" \
                        --graph_type "${graph_type}" \
                        --loss_type "${LOSS_TYPE}" \
                        --noise "${noise_type}" \
                        --prior_type "${prior_type}" \
                        --prior_rate "${PRIOR_RATE}" \
                        > "${log_file}" 2>&1

                    if [[ ! -f "${result_file}" ]]; then
                        echo "Expected result file was not created: ${result_file}" >&2
                        exit 1
                    fi
                done
            done
        done
    done
done

echo "All linear experiments completed."
