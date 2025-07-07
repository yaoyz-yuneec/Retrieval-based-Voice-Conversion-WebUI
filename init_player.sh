#!/bin/bash

echo "\$0: $0"
pause() {
    read -p "按任意键继续..."
}

# 检查是否被直接执行（而不是 source）
if [[ "$0" == "-bash" || "$0" == "-zsh" || "$0" == "-sh" || "$0" == "/bin/bash" || "$0" == "bash" ]]; then
    # 如果是被 source 加载，$0 通常是 shell 名称（如 bash、zsh）
    :
else
    echo "错误：此脚本必须用 'source' 加载，不能直接执行！" >&2
    echo "请使用： source $0" >&2
    pause
    exit 1
fi

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
# # 获取当前脚本的绝对路径，并解析所有符号链接
# # This is the most robust way to get the script's directory,
# # handling both direct execution and sourcing, as well as symlinks.
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "SCRIPT_DIR: $SCRIPT_DIR"
#. "${SCRIPT_DIR}/play_utils.sh"
#mapfile -t index_array < <(pacmd list-sinks | grep -B 1 -E "name: <alsa_output.usb-C-Media_Electronics_Inc|name: <alsa_output.usb-Generic_BT66_"  | grep 'index:' | awk -F: '{print $2}')
#index_value=${index_array[0]}
#echo "第一个索引值: ${index_value}"

# 查找包含"Server String:"的最后一行
last_server_line=$(grep "Server String:" /home/unix_ai/log/media_pkg/audio_srv_wait_ok.log | tail -n 1)

# 提取路径部分（如 /tmp/pulse-2ReOKTQ7JIzk/native）
pulse_path=$(echo "$last_server_line" | awk -F': ' '{print $2}')

# 设置环境变量并导出
if [ -n "$pulse_path" ]; then
    PULSE_RUNTIME_PATH=$(dirname "$pulse_path")
    PULSE_SERVER="unix:$pulse_path"  # 格式化为 unix:<path>

    export PULSE_RUNTIME_PATH
    export PULSE_SERVER

    echo "Exported:"
    echo "export PULSE_RUNTIME_PATH=$PULSE_RUNTIME_PATH"
    echo "export PULSE_SERVER=$PULSE_SERVER"
else
    echo "Error: No valid Server String found in the file."
    exit 1
fi
# pacmd set-default-sink $index_value
pactl info

