#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
data=(
    ### Real-World Set
    # "darkface"
    "darkface496"
    # "exdark"
    "exdark1200"
    "lolistreetval"
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
    --task "detect" \
    --mode "predict" \
    --arch "dfine" \
    --model "dfine_s" \
    --data "${data_str}" \
    --benchmark \
    --save-result \
    --save-image \
    --save-debug \
    --use-fullname \
    --save-nearby \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
