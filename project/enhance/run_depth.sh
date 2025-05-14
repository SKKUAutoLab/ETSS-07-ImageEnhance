#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
data=(

)
data_str=$(printf "%s, " "${data[@]}")
data_str=${data_str%, }  # Remove trailing ", "

# ----- Directory -----
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
root_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"

# ----- Main -----
cd "${runml_dir}" || exit
python -W ignore main.py \
    --root "${current_dir}" \
    --task "depth" \
    --mode "predict" \
    --arch "depth_anything_v2" \
    --model "depth_anything_v2_vitb" \
    --config 0 \
    --data "${data_str}" \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
