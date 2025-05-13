#!/bin/bash

# sudo chmod +x install.sh
# ./install.sh

clear
echo "$HOSTNAME"

# ----- Globals -----
# Directory
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
root_dir=$current_dir

# ----- Utils -----
check_gui_support() {
    if [ -n "$DISPLAY" ]; then
        # echo "GUI supported: X11 display server detected (DISPLAY=$DISPLAY)" >&2
        # echo "x11"
        return 0
    elif [ -n "$WAYLAND_DISPLAY" ]; then
        # echo "GUI supported: Wayland display server detected (WAYLAND_DISPLAY=$WAYLAND_DISPLAY)" >&2
        # echo "wayland"
        return 0
    else
        # echo "GUI not supported: No display server detected." >&2
        # echo "none"
        return 1
    fi
}

check_cuda() {
    if command -v nvcc >/dev/null 2>&1; then
        # echo "CUDA is installed. Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c 2-)"
        return 0
    elif command -v nvidia-smi >/dev/null 2>&1; then
        # echo "NVIDIA driver is installed. CUDA Version: $(nvidia-smi | grep CUDA | awk '{print $6}')"
        return 0
    else
        # echo "CUDA is not installed or not detected."
        return 1
    fi
}

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

add_bash_profile_lines() {
    local lines=("$@")
    local bash_profile="$HOME/.bash_profile"

    echo "Checking and adding lines to $bash_profile..." >&2
    for line in "${lines[@]}"; do
        if grep -Fx "$line" "$bash_profile" >/dev/null; then
            echo "Line already exists: $line" >&2
        else
            echo "Adding line: $line" >&2
            echo "$line" >> "$bash_profile" || {
                echo "Error: Failed to add line: $line" >&2
                echo "failed"
                return 1
            }
        fi
    done
}

get_env_yaml_path() {
    # echo -e "\nGetting environment YAML path"
    if check_cuda; then
        echo "${root_dir}/env/cuda.yaml"
    else
        echo "${root_dir}/env/cpu.yaml"
    fi
}

# ----- System -----
update_conda_channels() {
    echo -e "\nAdding 'conda' channels"
    conda config --append channels conda-forge
    conda config --append channels nvidia
    conda config --append channels pytorch
}

update_base_env() {
    echo -e "\nUpdating 'base' environment"
    conda update -n base -c defaults conda --y
    conda update --a --y
    pip install --upgrade pip poetry
}

install_ffmpeg() {
    echo -e "\nInstalling ffmpeg"
    case "$OSTYPE" in
        linux*)
            if sudo -n true 2>/dev/null; then
                sudo apt-get install ffmpeg
                sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
            else
                apt-get install ffmpeg
                apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
            fi
            ;;
        darwin*)
            brew install ffmpeg
            ;;
        win*)
            echo -e "\nWindows"
            ;;
        msys*)
            echo -e "\nMSYS / MinGW / Git Bash"
            ;;
        cygwin*)
            echo -e "\nCygwin"
            ;;
        bsd*)
            echo -e "\nBSD"
            ;;
        solaris*)
            echo -e "\nSolaris"
            ;;
        *)
            echo -e "\nunknown: $OSTYPE"
            ;;
    esac
}

install_imagemagick() {
    echo -e "\nInstalling imagemagick"
    case "$OSTYPE" in
        linux*)
            if sudo -n true 2>/dev/null; then
                sudo apt-get install imagemagick
            else
                apt-get install imagemagick
            fi
            ;;
        darwin*)
            brew install imagemagick
            ;;
        win*)
            echo -e "\nWindows"
            ;;
        msys*)
            echo -e "\nMSYS / MinGW / Git Bash"
            ;;
        cygwin*)
            echo -e "\nCygwin"
            ;;
        bsd*)
            echo -e "\nBSD"
            ;;
        solaris*)
            echo -e "\nSolaris"
            ;;
        *)
            echo -e "\nunknown: $OSTYPE"
            ;;
    esac
}

setup_resilio_sync() {
    rsync_dir="${root_dir}/.sync"
    mkdir -p "${rsync_dir}"
    cp "${root_dir}/env/IgnoreList" "${rsync_dir}/IgnoreList"
    echo -e "... Done"
}

update_system() {
    update_conda_channels
    update_base_env
    install_ffmpeg
    install_imagemagick
    setup_resilio_sync
}

# ----- Environment -----
create_mon_env_linux() {
    echo -e "\nCreating 'mon' environment:"
    # Install gcc and g++
    if sudo -n true 2>/dev/null; then
        sudo apt-get install gcc g++
    else
        apt-get install gcc g++
    fi
    # Create `mon` env
    env_yaml_path=$(get_env_yaml_path)
    conda env create -f "${env_yaml_path}"
    # Modify .bashrc
    bashrc_lines=(
        # "cd '${root_dir}'"
        "conda activate mon"
    )
    add_bashrc_lines "${bashrc_lines[@]}"
    source ~/.bashrc
    # Cleanup
    rm -rf $CONDA_PREFIX/lib/python3.12/site-packages/cv2/qt/plugins
    echo -e "... Done"
}

create_mon_env_darwin() {
    echo -e "\nCreating 'mon' environment:"
    # Must be called before installing PyTorch Lightning
    export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
    export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
    # Create `mon` env
    env_yaml_path=$(get_env_yaml_path)
    conda env create -f "${env_yaml_path}"
    # Modify .bash_profile
    bash_profile_lines=(
        # "cd '${root_dir}'"
        "conda activate mon"
    )
    add_bash_profile_lines "${bash_profile_lines[@]}"
    source ~/.bash_profile
    # Cleanup
    rm -rf $CONDA_PREFIX/lib/python3.12/site-packages/cv2/qt/plugins
    echo -e "... Done"
}

create_mon_env() {
    case "$OSTYPE" in
        linux*)
            create_mon_env_linux
            ;;
        darwin*)
            create_mon_env_darwin
            ;;
        win*)
            echo -e "\nWindows"
            ;;
        msys*)
            echo -e "\nMSYS / MinGW / Git Bash"
            ;;
        cygwin*)
            echo -e "\nCygwin"
            ;;
        bsd*)
            echo -e "\nBSD"
            ;;
        solaris*)
            echo -e "\nSolaris"
            ;;
        *)
            echo -e "\nunknown: $OSTYPE"
            ;;
    esac
}

install_mon_package() {
    create_mon_env

    echo -e "\nInstall 'mon' library"
    eval "$(conda shell.bash hook)"
    conda activate mon
    rm -rf poetry.lock
    if check_gui_support; then
        poetry install --extras "docs gui"
    else
        poetry install --extras "docs"
    fi
    rm -rf poetry.lock
    conda update --a --y
    conda clean  --a --y
}

# ----- Main -----
update_system
install_mon_package

# ----- Done -----
exit 0
