#writefile train_blue.sh
#!/usr/bin/env bash

# ★ 建立一個全新的測試資料夾，避免覆蓋之前的 Baseline
CKPT_DIR="/content/drive/MyDrive/Colab Notebooks/checkpoint_test10"
mkdir -p "$CKPT_DIR"

# 自動尋找最新的 Epoch 存檔
LATEST_CKPT=$(ls -v "$CKPT_DIR"/[0-9]*.pt.tar 2>/dev/null | tail -n 1)

if [ -f "$LATEST_CKPT" ]; then
    echo "🚀 發現中斷的存檔：$LATEST_CKPT"
    echo "🔄 啟動自動接關模式 (Auto-Resume)..."
    python trainnew_blue.py --gpus 0 --epochs 10 --checkpoint "$CKPT_DIR" --batch-size 2 --num-workers 4 --trainer_type origin --resume "$LATEST_CKPT"
else
    echo "🌱 啟動全新 1/2 資料量測試 (Epoch 10, Batch Size 2)..."
    python trainnew_blue.py --gpus 0 --epochs 10 --checkpoint "$CKPT_DIR" --batch-size 2 --num-workers 4 --trainer_type origin
fi