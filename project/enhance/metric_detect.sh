#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
arch="retinexnet"
model="retinexnet_lolv1"
detector="dfine_s_coco80"
data=(
    ### Real-World Set
    # "darkface"
    # "exdark"
    "exdark1200"
    "lolistreetval"
)
device="cuda:0"
bbox_format="yolo"

# ----- Globals -----
# Directory
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
root_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"

# ----- Main -----
cd "${runml_dir}" || exit
for (( i=0; i<${#data[@]}; i++ )); do
    # Input
    input_dir="${current_dir}/run/predict/${arch}/${model}/${data[i]}/pred"
    label_dir="${current_dir}/run/predict/${arch}/${model}/${data[i]}/pred_${detector}/label"
    if ! [ -d "${label_dir}" ]; then
        label_dir="${current_dir}/run/predict/${arch}/${model}/${data[i]}/pred_${detector}"
    fi
    # input_json="${label_dir}.json"
    input_json="${current_dir}/run/predict/${arch}/${model}/${data[i]}/pred_${detector}.json"

    # Target
    if [ "${data[i]}" == "exdark" ]; then
        target_json="${current_dir}/data/exdark/test/test.json"
        remap_classes="${current_dir}/data/exdark/remap_coco802exdark.yaml"
    elif [ "${data[i]}" == "exdark1200" ]; then
        target_json="${current_dir}/data/exdark/test1200/test1200.json"
        remap_classes="${current_dir}/data/exdark/remap_coco802exdark.yaml"
    elif [ "${data[i]}" == "lolistreetval" ]; then
        target_json="${current_dir}/data/lolistreet/val/ref.json"
        remap_classes=""
    else
        target_json="${current_dir}/data/${data[i]}/test/test.json"
        remap_classes=""
    fi

    # Measure COCO
    python -W ignore metric_coco.py \
        --input-dir "${input_dir}" \
        --label-dir "${label_dir}" \
        --input-json "${input_json}" \
        --target-json "${target_json}" \
        --result-file "${current_dir}" \
        --remap-classes "${remap_classes}" \
        --arch "${arch}" \
        --model "${model}" \
        --data "${data[i]}" \
        --device "${device}" \
        --bbox-format "${bbox_format}" \
        --exist-ok

done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
