#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
data="dicm, fusion, lime, mef, npe, vv, lolv1, lolv2real, lolv2syn, fivek, sice, sicegrad, sicemix, sidsony, darkcityscapes, darkface, exdark, lolistreettest, lolistreetval, nightcity"

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
    --task "lle" \
    --mode "predict" \
    --data "${data}" \
    --benchmark \
    --save-image \
    --save-debug \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
