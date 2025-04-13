#!/bin/bash
# 设置日志文件路径
LOG_DIR="logs"
mkdir -p $LOG_DIR  # 确保日志目录存在
LOG_FILE="$LOG_DIR/train_$(date +"%Y%m%d_%H%M%S").log"

# 运行 Python 代码并将日志写入文件
#python xxx.py > "$LOG_FILE" 2>&1
#python main_fed.py --asynchronous --time_weighted  --num_users 10 --active_users 1 --lr 0.005 --epochs 300 --all_clients --keep_log \
python main_fed.py --asynchronous --time_weighted --epochs 200 --local_ep 5 --lr 0.01  --keep_log \
| tee -a "$LOG_FILE"

# 打印日志路径
echo "Log saved to $LOG_FILE"
