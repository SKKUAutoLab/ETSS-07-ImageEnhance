#!/bin/bash

clear
echo "$HOSTNAME"

# ----- Input -----
option=${1:-"start"}
read -e -i "$option" -p "Option [install, start]: " option

# ----- Globals -----
# Directory
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")  # mon/tool/
root_dir=$(dirname "$current_dir")      # mon/
xanylabeling_dir="${current_dir}/X-AnyLabeling"

# ----- Check -----
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

# ----- Setup -----
install_xanylabeling() {
  echo -e "\nInstall X-AnyLabeling"

  # Download repo
  if [ ! -d "$xanylabeling_dir" ]; then
    git clone https://github.com/CVHub520/X-AnyLabeling.git
  fi
  cd "${xanylabeling_dir}" || exit

  # Create conda environment
  conda create --name xanylabeling python=3.9 --y
  eval "$(conda shell.bash hook)"
  conda activate xanylabeling
  pip install -U pip
  if check_cuda; then
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
  fi

  case "$OSTYPE" in
    linux*)
      pip install -r requirements-dev.txt
      if sudo -n true 2>/dev/null; then
          sudo apt-get install libxcb-xinerama0
      else
          apt-get install libxcb-xinerama0
      fi
      ;;
    darwin*)
        pip install -r requirements-macos-dev.txt
        ;;
    *)
        echo -e "\nunknown: $OSTYPE"
        ;;
  esac
}

# ----- Main -----
# Install
if [ "${option}" == "install" ]; then
    install_xanylabeling
fi

# Start
if [ "${option}" == "start" ]; then
    eval "$(conda shell.bash hook)"
    conda activate xanylabeling
    cd "${xanylabeling_dir}" || exit
    python anylabeling/app.py
fi

# ----- Done -----
exit 0
