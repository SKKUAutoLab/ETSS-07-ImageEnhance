#!/bin/bash

clear
echo "$HOSTNAME"

# ----- Input -----
option=${1:-"install"}
read -e -i "$option" -p "Option [install, enable, disable, stop, start]: " option

# ----- Globals -----
# Directory
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
root_dir=$current_dir

# ----- Setup -----
install() {
    service_file="${root_dir}/env/resilio-sync.service"
    target_file="/usr/lib/systemd/user/resilio-sync.service"
    cp "${service_file}" "${target_file}"
}

# ----- Main -----
# Install
if [ "${option}" == "install" ]; then
    install
fi

# Enable/Disable/Start/Stop
case "${option}" in
    enable)
        systemctl --user enable resilio-sync
        ;;
    disable)
        systemctl --user disable resilio-sync
        ;;
    stop)
        systemctl --user stop resilio-sync
        ;;
    start)
        systemctl --user start resilio-sync
        ;;
    *)
        echo "Invalid option: $option"
        ;;
esac

# ----- Done -----
exit 0
