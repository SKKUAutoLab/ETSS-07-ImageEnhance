#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
data=(
    ### Unpaired Set
    "dicm"
    "lime"
    "mef"
    "npe"
    "vv"
    ### LOLs Set
    "lolv1"
    "lolv2real"
    "lolv2syn"
    ### FiveK Set
    "fivek"
    #"fiveka"
    #"fivekb"
    #"fivekc"
    #"fivekd"
    #"fiveke"
    ### SICE Set
    "sice"
    "sicegrad"
    "sicemix"
    ### Camera-Specific Set
    "sidsony"
    ### Real-World Set
    "darkcityscapes"
    # "darkface"
    "darkface496"
    # "exdark"
    "exdark1200"
    "lolistreettest"
    "lolistreetval"
    "nightcity"
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
    --task "lle" \
    --mode "predict" \
    --data "${data_str}" \
    --benchmark \
    --save-result \
    --save-image \
    --save-debug \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
