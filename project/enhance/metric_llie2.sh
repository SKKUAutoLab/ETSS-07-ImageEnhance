#!/bin/bash
echo "$HOSTNAME"
clear

# Directories
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
mon_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"
data_dir="${mon_dir}/data/enhance"

# Input
arch="zero_mie"
model="zero_mie_ms"
data=(
    ### Unpaired Set
    "dicm"
    "lime"
    "mef"
    "npe"
    "vv"
    ### LOLs Set
    "lol_v1"
    "lol_v2_real"
    "lol_v2_synthetic"
    ### Multiple Exposure Set
    "fivek_e"
    "sice"
    "sice_grad"
    "sice_mix_v2"
    ### Camera-Specific Set
    # "sid_sony"
    ### Real-World Set
    # "darkcityscapes"
    # "darkface"
    # "exdark"
    # "loli_street_val"
    "loli_street_test"
    "nightcity"
)
device="cuda:0"

# Run
cd "${runml_dir}" || exit
for (( i=0; i<${#data[@]}; i++ )); do
    # Input
    input_dir="${data_dir}/#predict/${arch}/${model}/${data[i]}"
    if ! [ -d "${input_dir}" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/${data[i]}"
    fi
    # Target
    if [ "${data[i]}" == "loli_street_val" ]; then
        target_dir="${data_dir}/loli_street/val/ref"
    elif [ "${data[i]}" == "loli_street_test" ]; then
        target_dir="${data_dir}/loli_street/test/ref"
    else
        target_dir="${data_dir}/${data[i]}/test/ref"
        if ! [ -d "${target_dir}" ]; then
            target_dir="${data_dir}/${data[i]}/val/ref"
        fi
    fi

    # Measure FR-IQA
    if [ -d "${target_dir}" ]; then
        python -W ignore metric.py \
          --input-dir "${input_dir}" \
          --target-dir "${target_dir}" \
          --result-file "${current_dir}" \
          --arch "${arch}" \
          --model "${model}" \
          --data "${data[i]}" \
          --device "${device}" \
          --imgsz 512 \
          --metric "psnr" \
          --metric "ssimc" \
          --metric "psnry" \
          --metric "ssim" \
          --metric "ms_ssim" \
          --metric "lpips" \
          --metric "brisque" \
          --metric "ilniqe" \
          --metric "niqe" \
          --metric "pi" \
          --backend "pyiqa" \
          --use-gt-mean
    # Measure NR-IQA
    else
        python -W ignore metric.py \
          --input-dir "${input_dir}" \
          --target-dir "${target_dir}" \
          --result-file "${current_dir}" \
          --arch "${arch}" \
          --model "${model}" \
          --data "${data[i]}" \
          --device "${device}" \
          --imgsz 512 \
          --metric "psnr" \
          --metric "ssimc" \
          --metric "psnry" \
          --metric "ssim" \
          --metric "ms_ssim" \
          --metric "lpips" \
          --metric "brisque" \
          --metric "ilniqe" \
          --metric "niqe" \
          --metric "pi" \
          --backend "pyiqa"
    fi

done

# Done
cd "${current_dir}" || exit
