#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
arch="retinexnet"
model="retinexnet_lolv1"
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
    #"fiveka"
    #"fivekb"
    "fivekc"
    #"fivekd"
    "fiveke"
    ### SICE Set
    "sice"
    "sicegrad"
    "sicemix"
    ### Camera-Specific Set
    "sidsony"
    ### Real-World Set
    "darkcityscapes"
    "darkface"
    # "exdark"
    "exdark1200"
    "lolistreettest"
    "lolistreetval"
    "nightcity"
)
device="cuda:0"

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
    if [ "${data[i]}" == "fiveka" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek/pred"
    elif [ "${data[i]}" == "fivekb" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek/pred"
    elif [ "${data[i]}" == "fivekc" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek/pred"
    elif [ "${data[i]}" == "fivekd" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek/pred"
    elif [ "${data[i]}" == "fiveke" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek/pred"
    else
        input_dir="${current_dir}/run/predict/${arch}/${model}/${data[i]}/pred"
    fi

    # Target
    if [ "${data[i]}" == "lolv2real" ]; then
        target_subdir="lolv2/real/test/ref"
    elif [ "${data[i]}" == "lolv2syn" ]; then
        target_subdir="lolv2/syn/test/ref"
    elif [ "${data[i]}" == "fiveka" ]; then
        target_subdir="fivek/test/ref_a"
    elif [ "${data[i]}" == "fivekb" ]; then
        target_subdir="fivek/test/ref_b"
    elif [ "${data[i]}" == "fivekc" ]; then
        target_subdir="fivek/test/ref_c"
    elif [ "${data[i]}" == "fivekd" ]; then
        target_subdir="fivek/test/ref_d"
    elif [ "${data[i]}" == "fiveke" ]; then
        target_subdir="fivek/test/ref_e"
    elif [ "${data[i]}" == "sice" ]; then
        target_subdir="sice/sice/test/ref"
    elif [ "${data[i]}" == "sicegrad" ]; then
        target_subdir="sice/grad/test/ref"
    elif [ "${data[i]}" == "sicemix" ]; then
        target_subdir="sice/mix/test/ref"
    elif [ "${data[i]}" == "sidsony" ]; then
        target_subdir="sid/sony/test/ref"
    elif [ "${data[i]}" == "exdark1200" ]; then
        target_subdir="sid/sony/test1200/ref"
    elif [ "${data[i]}" == "lolistreetval" ]; then
        target_subdir="lolistreet/val/ref"
    elif [ "${data[i]}" == "lolistreettest" ]; then
        target_subdir="lolistreet/test/ref"
    else
        target_subdir="${data[i]}/test/ref"
    fi
    target_dir="${current_dir}/data/${target_subdir}"
    if ! [ -d "${target_dir}" ]; then
        target_dir="${root_dir}/data/enhance/${target_subdir}"
    fi

    # Measure FR-IQA
    if [ -d "${target_dir}" ]; then
        python -W ignore metric_iqa.py \
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
        python -W ignore metric_iqa.py \
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

# ----- Done -----
cd "${current_dir}" || exit
exit 0
