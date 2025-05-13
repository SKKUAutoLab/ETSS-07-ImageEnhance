#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
data="rain100, rain100l, rain100h, rain800, rain1200, rain1400, rain2800, raincityscapes, gtrain"

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
    --task "derain" \
    --mode "predict" \
    --data "${data}" \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
