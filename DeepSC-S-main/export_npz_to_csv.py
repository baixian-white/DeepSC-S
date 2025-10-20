# -*- coding: utf-8 -*-
"""
功能：
--------
批量读取 demo_intermediate/ 目录下的所有 .npz 文件，
将每个文件中的所有数组（feat_sem, batch_mean, batch_var 等）
完全展开（flatten 成一维向量），并保存为 CSV 文件。

用法：
--------
python export_npz_to_csv.py \
    --input_dir /workspace/demo_intermediate \
    --out_dir /workspace/npz_csv
"""

import os
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# 1) 解析命令行参数
# -------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="将 .npz 文件中的特征展开并导出为 CSV（Flatten 全展开）")
    parser.add_argument("--input_dir", required=True, help="包含 .npz 文件的目录")
    parser.add_argument("--out_dir", default="./npz_csv", help="CSV 文件输出目录")
    return parser.parse_args()

# -------------------------------------------------------------
# 2) 主转换函数：读取单个 npz 文件并展开写入 CSV
# -------------------------------------------------------------
def export_npz_to_csv(npz_path, out_dir):
    """
    将单个 .npz 文件中所有数组 flatten 成一维并写入 CSV。
    每个 key（feat_sem / batch_mean 等）各生成一个 CSV 文件。
    """
    base = os.path.splitext(os.path.basename(npz_path))[0]  # 文件名（不带扩展名）
    os.makedirs(out_dir, exist_ok=True)                     # 确保输出目录存在

    # 加载 npz 文件（类似字典，键是数组名）
    data = np.load(npz_path)
    print(f"[LOAD] {npz_path} -> keys={data.files}")

    # 遍历每个数组
    for key in data.files:
        arr = data[key]                      # 取出 numpy 数组
        arr_shape = arr.shape                # 获取数组形状（例如 (32,32,32,128)）
        print(f"  - {key}: shape={arr_shape}, dtype={arr.dtype}")

        # flatten: 将多维数组拉平成一维
        flat = arr.flatten()

        # 转为 pandas DataFrame（方便保存为 CSV）
        df = pd.DataFrame({"value": flat})

        # 拼接输出路径
        csv_name = f"{base}_{key}.csv"
        csv_path = os.path.join(out_dir, csv_name)

        # 写入 CSV（不保存行索引）
        df.to_csv(csv_path, index=False)

        # 打印保存信息
        print(f"[SAVE] {csv_path} ({len(df)} rows)")

# -------------------------------------------------------------
# 3) 主程序入口：批量处理目录下的所有 npz
# -------------------------------------------------------------
def main():
    args = parse_args()  # 解析参数
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".npz")]  # 收集所有 npz 文件
    if not files:
        raise FileNotFoundError(f"目录 {args.input_dir} 下未找到 .npz 文件")

    # 逐个文件处理
    for fname in sorted(files):
        fpath = os.path.join(args.input_dir, fname)
        export_npz_to_csv(fpath, args.out_dir)

    print("\n✅ 全部转换完成！CSV 文件保存在：", args.out_dir)

# -------------------------------------------------------------
if __name__ == "__main__":
    main()
