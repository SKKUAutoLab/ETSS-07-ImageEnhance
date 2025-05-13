#!/bin/bash

# ubuntu 22.04, nvidia-driver-570, cuda-toolkit-12-6

clear
echo "$HOSTNAME"

# ----- Globals -----
# Directory
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
root_dir=$current_dir

# ----- Utils -----
add_bashrc_lines() {
    local lines=("$@")
    local bashrc="$HOME/.bashrc"

    echo "Checking and adding lines to $bashrc..." >&2
    for line in "${lines[@]}"; do
        if grep -Fx "$line" "$bashrc" >/dev/null; then
            echo "Line already exists: $line" >&2
        else
            echo "Adding line: $line" >&2
            echo "$line" >> "$bashrc" || {
                echo "Error: Failed to add line: $line" >&2
                echo "failed"
                return 1
            }
        fi
    done
}

# ----- Step 1: Upgrade Ubuntu -----
upgrade_ubuntu() {
    echo -e "\nUpgrading Ubuntu"
    sudo apt update
    sudo apt upgrade
    sudo apt install gcc g++
}

# ----- Step 2: Install NVIDIA Driver -----
install_nvidia_driver() {
    echo -e "\nInstalling NVIDIA driver"
    sudo apt install ubuntu-drivers-common
    sudo ubuntu-drivers devices
    sudo apt install nvidia-driver-570
    # sudo reboot now
}

# ----- Step 3: Install CUDA Toolkit -----
install_cuda_toolkit() {
    echo -e "\nInstalling CUDA Toolkit"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-6
    # Modify .bashrc
    bashrc_lines=(
        "export PATH=/usr/local/cuda/bin\${PATH:+:\${PATH}}"
        "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
    )
    add_bashrc_lines "${bashrc_lines[@]}"
    source ~/.bashrc
    # Manually add to ~/.bashrc
    # sudo nano ~/.bashrc
    # echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}"
    # echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    # source ~/.bashrc
    # sudo reboot now
}

# ----- Main -----
upgrade_ubuntu
install_nvidia_driver
install_cuda_toolkit

# ----- Done -----
exit 0
