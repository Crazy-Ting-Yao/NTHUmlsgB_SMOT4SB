#!/bin/bash

# 設定 Python 腳本名稱
PYTHON_SCRIPT="track.py"

# 設定目標資料夾
DATA_DIR="/home/jingxunlin/SMOT/dataset/SMOT4SB/pub_test/"

# 確保資料夾存在
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: 資料夾 $DATA_DIR 不存在！"
    exit 1
fi

# 逐一處理資料夾內的檔案
for dir in "$DATA_DIR"/*; do
    if [ -d "$dir" ]; then
        folder_name=$(basename "$dir")
        echo "Processing: $folder_name"
        python3 "$PYTHON_SCRIPT" "$folder_name"
    fi
done

echo "所有檔案處理完成！"
python3 "../MVA2025-SMOT4SB/scripts/create_submission.py" -i "pub_test/"