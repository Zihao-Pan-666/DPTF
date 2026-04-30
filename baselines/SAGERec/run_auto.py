import subprocess
import os
import time

# 1. 定义要顺序执行的任务列表 (数据集, 模型名称, 损失模式, 特定 lambda_g)
tasks = [
    {"dataset": "amazon_movies_and_tv", "model": "bert4rec", "loss": "sage", "lambda_g": "0.05"},
    {"dataset": "amazon_movies_and_tv", "model": "sasrec", "loss": "sage", "lambda_g": "0.05"},
    {"dataset": "amazon_movies_and_tv", "model": "gru4rec", "loss": "sage", "lambda_g": "0.05"},
]

# 2. 定义基础参数 (包含你找到的最佳 SAGERec 辅助超参数)
base_cmd = [
    "python", "-u", "sagerec_main.py",
    "--early_stop_criterion", "zero_shot",
    "--early_stop_patience", "10",
    "--gamma_g", "0.1",   # 最佳 Gamma
    "--beta_id", "0.1",   # 最佳 Beta
    "--tau", "0.2",       # 最佳 Tau
    "--force_training"
]

# 创建日志文件夹
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print("===================================================")
print("🚀 开始顺序执行批量实验 (更新ID符号)...")
print("===================================================")

start_total_time = time.time()

# 3. 循环执行每个任务
for task in tasks:
    dataset = task["dataset"]
    model = task["model"]
    loss_mode = task["loss"]
    lambda_g = task["lambda_g"]

    # 优化日志文件名，加入数据集名称防止覆盖
    log_file_path = os.path.join(log_dir, f"run_{dataset}_{model}_{loss_mode}.log")
    print(f"\n>>> 当前任务: 数据集=[{dataset}], 模型=[{model}], 模式=[{loss_mode}], λ=[{lambda_g}]")
    print(f">>> 日志文件: {log_file_path}")

    # 动态组装完整命令
    current_cmd = base_cmd + [
        "--dataset_name", dataset,
        "--model_name", model,
        "--loss_mode", loss_mode,
        "--lambda_g", lambda_g
    ]

    # 运行子进程
    with open(log_file_path, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            current_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",  # 确保不报 GBK 解码错误
            errors="ignore",  # 忽略进度条带来的乱码
            bufsize=1
        )

        # 实时读取并过滤输出
        for line in process.stdout:
            # 核心过滤逻辑：只要包含 tqdm 进度条特征（%| 或 ?it/s），就丢弃它
            if "%|" not in line and "?it/s" not in line:
                f.write(line)
                f.flush()
                # 为了不让屏幕显得死机，在屏幕上打印重要信息（Epoch汇总、INFO信息、最终结果）
                if "Epoch" in line or "FINAL" in line or "[INFO]" in line or ">>>" in line:
                    print(f"  {line.strip()}")

    process.wait()
    print(f"<<< 任务 [{dataset}-{model}-{loss_mode}] 执行完毕。")
    print("-" * 60)

end_total_time = time.time()
hours, rem = divmod(end_total_time - start_total_time, 3600)
minutes, seconds = divmod(rem, 60)

print("🎉 所有指定实验已按顺序执行完毕！")
print(f"⏳ 总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")