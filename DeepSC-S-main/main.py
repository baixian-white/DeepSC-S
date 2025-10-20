# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:41:39 2020

@author: Zhenzi Weng
说明：
- 端到端 DeepSC-S 训练脚本（TensorFlow 2.x）
- 输入：TFRecords（每条样本是一段 8kHz、长度为 16384 点的波形窗口）
- 模型：语义编码器 -> 信道编码器 ->（Rayleigh+AWGN 信道）-> 信道解码器 -> 语义解码器
- 损失：MSE（重建波形 vs 原波形）
"""

from __future__ import absolute_import          # 兼容 Python2/3 的绝对导入行为
from __future__ import division                 # 使用真实除法（/ 得到浮点），避免 Python2 的整除陷阱
from __future__ import print_function           # 使用 Python3 风格的 print 函数

import os                                       # 操作系统相关（环境变量/路径/文件）
import time                                     # 计时（每个 epoch 的耗时）
import argparse                                 # 命令行参数解析
import tensorflow as tf                         # 深度学习框架
import numpy as np                              # 数值计算
import scipy.io as sio                          # 读写 .mat（保存 loss 曲线）
from models import (
    sem_enc_model, chan_enc_model, Chan_Model, chan_dec_model, sem_dec_model
)  # 自定义模块工厂：语义编码器/信道编码器/信道模型/信道解码器/语义解码器
# ==== 新增：一次训练 = 一个独立目录，集中保存输出 ====
import json, datetime
from pathlib import Path
import matplotlib.pyplot as plt


# 查询并打印 CPU 核心数（用于 tf.data 并行 map 的 num_parallel_calls）
num_cpus = os.cpu_count()
print("Number of CPU cores is", num_cpus)

# 指定可见 GPU（若只有一张卡，也写 "0" 即可；如需 CPU 训练可注释掉此行）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 让显存按需增长：避免一次性占满显存导致 OOM 或与其他进程冲突
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
def save_run_outputs(train_loss_epoch_list, valid_loss_epoch_list, args_dict, extra=None):
    """
    将一次训练的结果保存到独立目录：trained_outputs/runs/<RUN_ID>/
    - train_loss.mat / valid_loss.mat（各自只有一个键）
    - train_loss.csv / valid_loss.csv
    - loss_curve.png
    - config.json（记录本次训练的全部参数）
    - 可选 extra（字典），保存你希望的任意辅助结果
    返回：run_dir（字符串路径）
    """
    # 以时间戳作为 run_id，确保每次训练独立
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path("/workspace/trained_outputs")
    run_dir = base_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) 保存 .mat（只有一个键，不会产生 train_loss_1 之类）
    sio.savemat(str(run_dir / "train_loss.mat"), {"train_loss": np.array(train_loss_epoch_list, dtype=np.float32)})
    if valid_loss_epoch_list is not None:
        sio.savemat(str(run_dir / "valid_loss.mat"), {"valid_loss": np.array(valid_loss_epoch_list, dtype=np.float32)})

    # 2) 保存 .csv（方便快速打开查看）
    np.savetxt(str(run_dir / "train_loss.csv"), np.array(train_loss_epoch_list, dtype=np.float32), delimiter=",")
    if valid_loss_epoch_list is not None:
        np.savetxt(str(run_dir / "valid_loss.csv"), np.array(valid_loss_epoch_list, dtype=np.float32), delimiter=",")

    # 3) 保存参数快照
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)

    # 4) 保存收敛曲线图（WSL/Docker 无 GUI 也没问题，因为是直接保存 PNG）
    try:
        plt.figure()
        plt.plot(train_loss_epoch_list, label="Train Loss")
        if valid_loss_epoch_list is not None:
            plt.plot(valid_loss_epoch_list, label="Valid Loss")
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title(f"Loss Curve ({run_id})")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(str(run_dir / "loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"[warn] 绘图保存失败（忽略）：{e}")

    # 5) 方便快速定位最新一次：写一个 latest 指针（符号链接或文本）
    try:
        latest_link = base_dir/ "runs" / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir, target_is_directory=True)
    except Exception:
        with open(base_dir / "runs" / "LATEST.txt", "w", encoding="utf-8") as f:
            f.write(str(run_dir))

    print(f"✅ 本次训练已保存到：{run_dir}")
    return str(run_dir)

###############    define global parameters    ###############
def parse_args():
    """
    定义命令行参数：
    - 帧参数（采样率/帧长/帧移/帧数）
    - 编码器/解码器通道配置
    - 训练/验证 TFRecords 路径
    - 训练超参（SNR、epoch、batch_size、学习率）
    """
    parser = argparse.ArgumentParser(description="semantic communication systems for speech transmission")

    # parameter of frame
    parser.add_argument("--sr", type=int, default=8000, help="sample rate for wav file")
    # 语音采样率（Hz），与数据处理脚本一致，默认 8kHz（对应电话带宽）
    parser.add_argument("--num_frame", type=int, default=128, help="number of frames in each batch")
    # 每个样本包含的帧数（F），默认 128（窗口里总共有 128 帧）
    parser.add_argument("--frame_size", type=float, default=0.016, help="time duration of each frame")
    # 每帧时长（秒），默认 0.016 s → 配合 sr=8000 得到 frame_length=128 采样点
    parser.add_argument("--stride_size", type=float, default=0.016, help="time duration of frame stride")
    # 帧移（秒），默认与帧长相同（无重叠）。如果小于 frame_size，则帧间有重叠

    # parameter of semantic coding and channel coding
    parser.add_argument("--sem_enc_outdims", type=list, default=[32, 128, 128, 128, 128, 128, 128],
                        help="output dimension of SE-ResNet in semantic encoder.")
    # 语义编码器里各级的通道配置（列表）：前两项给下采样卷积，后面给若干 SEResNet 模块
    parser.add_argument("--chan_enc_filters", type=list, default=[128],
                        help="filters of CNN in channel encoder.")
    # 信道编码器卷积的输出通道列表（通常只用第一项作为该层输出通道数）
    parser.add_argument("--chan_dec_filters", type=list, default=[128],
                        help="filters of CNN in channel decoder.")
    # 信道解码器卷积的输出通道列表（同理，通常只用第一项）
    parser.add_argument("--sem_dec_outdims", type=list, default=[128, 128, 128, 128, 128, 128, 32],
                        help="output dimension of SE-ResNet in semantic decoder.")
    # 语义解码器里各级（含反卷积前的 SEResNet 模块）的通道配置（最后两项常用于反卷积）

    # path of tfrecords files
    parser.add_argument("--trainset_tfrecords_path", type=str, default="./data_tfrecords_8k/trainset.tfrecords",
                        help="tfrecords path of trainset.")
    # 训练集 TFRecords 文件路径（需改成你真实路径）
    parser.add_argument("--validset_tfrecords_path", type=str, default="./data_tfrecords_8k/validset.tfrecords",
                        help="tfrecords path of validset.")
    # 验证集 TFRecords 文件路径（需改成你真实路径）

    # parameter of wireless channel
    parser.add_argument("--snr_train_dB", type=int, default=8, help="snr in dB for training.")
    # 训练阶段的信噪比（dB）。将被换算成噪声标准差 std 喂给信道模型

    # epoch and learning rate
    parser.add_argument("--num_epochs", type=int, default=1000, help="training epochs.")
    # 训练的 epoch 数（默认 1000）
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")
    # mini-batch 大小（默认 32），预处理脚本已将样本总数补齐为 batch 的整数倍
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate.")
    # 学习率（RMSprop 的基础步长）

    args = parser.parse_args()
    return args


args = parse_args()
print("Called with args:", args)

# 推导帧长/帧移（单位：采样点）
frame_length = int(args.sr * args.frame_size)   # 8000 * 0.016 = 128
stride_length = int(args.sr * args.stride_size) # 8000 * 0.016 = 128（无重叠）

if __name__ == "__main__":

    ###############    define system model    ###############
    # 1) 语义编码器
    sem_enc = sem_enc_model(frame_length, stride_length, args)
    print(sem_enc.summary(line_length=160))

    # 2) 信道编码器1
    chan_enc = chan_enc_model(frame_length, args)
    print(chan_enc.summary(line_length=160))

    # 3) 信道层（Rayleigh + AWGN + 理想均衡）
    chan_layer = Chan_Model(name="Channel_Model")

    # 4) 信道解码器
    chan_dec = chan_dec_model(frame_length, args)
    print(chan_dec.summary(line_length=160))

    # 5) 语义解码器
    sem_dec = sem_dec_model(frame_length, stride_length, args)
    print(sem_dec.summary(line_length=160))

    # 收集全部可训练权重（不含信道层）
    weights_all = (
        sem_enc.trainable_weights +
        chan_enc.trainable_weights +
        chan_dec.trainable_weights +
        sem_dec.trainable_weights
    )

    # 定义 MSE 损失
    mse_loss = tf.keras.losses.MeanSquaredError(name="mse_loss")

    # 定义优化器（可替换为 Adam/SGD 等）
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr)

    ###############    define train step and valid step    ###############
    @tf.function
    def train_step(_input, std):
        """
        单个训练 step：
        - 前向：语义编码 -> 信道编码 -> 信道 -> 信道解码 -> 语义解码
        - 计算 MSE
        - 反向传播 + 参数更新
        """
        std = tf.cast(std, dtype=tf.float32)
        with tf.GradientTape() as tape:
            _output, batch_mean, batch_var = sem_enc(_input)      # 语义编码
            _output = chan_enc(_output)                           # 信道编码
            _output = chan_layer(_output, std)                    # 通过信道
            _output = chan_dec(_output)                           # 信道解码
            _output = sem_dec([_output, batch_mean, batch_var])   # 语义解码
            loss_value = mse_loss(_input, _output)                # 重建损失

        grads = tape.gradient(loss_value, weights_all)             # 求梯度
        optimizer.apply_gradients(zip(grads, weights_all))         # 更新参数
        return loss_value

    @tf.function
    def valid_step(_input, std):
        """验证 step：仅前向与损失计算，不做反向/更新。"""
        std = tf.cast(std, dtype=tf.float32)
        _output, batch_mean, batch_var = sem_enc(_input)
        _output = chan_enc(_output)
        _output = chan_layer(_output, std)
        _output = chan_dec(_output)
        _output = sem_dec([_output, batch_mean, batch_var])
        loss_value = mse_loss(_input, _output)
        return loss_value

    ###############    map function to read tfrecords    ###############
    @tf.function
    def map_function(example):
        """
        解析一条 TFRecord 样本（字节 → 波形张量）
        TFRecord schema:
            - "wav_raw": bytes，长度应为 16384 * 2（int16）
        """
        feature_map = {"wav_raw": tf.io.FixedLenFeature([], tf.string)}
        parsed_example = tf.io.parse_single_example(example, features=feature_map)
        wav_slice = tf.io.decode_raw(parsed_example["wav_raw"], out_type=tf.int16)  # int16 [16384]
        wav_slice = tf.cast(wav_slice, tf.float32) / 2**15                          # 归一化到 ~[-1,1]
        return wav_slice

    ###################    create folder to save data    ###################
    # 建议将保存路径放在 /workspace（你把主机 E:\ 挂载到了 /workspace），这样容器退出也能保留
    common_dir = "/workspace/trained_outputs/"
    saved_model = os.path.join(common_dir, "saved_model/")
    os.makedirs(saved_model, exist_ok=True)

    # 训练/验证 loss 的保存目录
    # train_loss_dir = os.path.join(common_dir, "train/")
    # os.makedirs(train_loss_dir, exist_ok=True)       # 目录已存在时不报错
    # train_loss_file = os.path.join(train_loss_dir, "train_loss.mat")
    train_loss_all = []                              # 用列表暂存每个 epoch 的逐 step 损失

    # valid_loss_dir = os.path.join(common_dir, "valid/")
    # os.makedirs(valid_loss_dir, exist_ok=True)
    # valid_loss_file = os.path.join(valid_loss_dir, "valid_loss.mat")
    valid_loss_all = []

    train_loss_epoch_means = []  # 仅保存每个 epoch 的平均损失（用于绘图/保存）
    valid_loss_epoch_means = []

    print("*****************   start train   *****************")

    # 将 SNR(dB) 换算为线性值，再推导复高斯噪声 std：
    # 若 E[|x|^2] ≈ 1，则 SNR ≈ 1 / (2 * std^2) => std = sqrt(1 / (2 * SNR))
    snr = pow(10, (args.snr_train_dB / 10))
    std = np.sqrt(1 / (2 * snr)).astype(np.float32)

    for epoch in range(args.num_epochs):
        ##########################    train    ##########################
        # 训练集数据管道
        trainset = tf.data.TFRecordDataset(args.trainset_tfrecords_path)          # 读 TFRecords
        trainset = trainset.map(map_function, num_parallel_calls=num_cpus)        # 解析 bytes→float
        trainset = trainset.shuffle(buffer_size=args.batch_size * 657,            # 近似全局洗牌
                                    reshuffle_each_iteration=True)
        trainset = trainset.batch(batch_size=args.batch_size, drop_remainder=False)  # 默认保留最后一小批
        trainset = trainset.prefetch(buffer_size=args.batch_size)                  # 预取提速

        train_loss_epoch = []             # 暂存本 epoch 每个 step 的 loss
        train_loss = 0.0                  # 累加平均
        start = time.time()               # 计时

        for step, _input in enumerate(trainset):
            loss_value = train_step(_input, std)
            loss_float = float(loss_value.numpy() if hasattr(loss_value, "numpy") else loss_value)
            train_loss_epoch.append(loss_float)
            train_loss += loss_float

        train_loss /= (step + 1)          # 本 epoch 平均训练损失
        train_loss_all.append(np.array(train_loss_epoch, dtype=np.float32))  # 记录曲线
        train_loss_epoch_means.append(float(train_loss))   # 每个 epoch 的平均训练损失


        print("train epoch {}/{}, train_loss = {:.06f}, time = {:.06f}".format(
            epoch + 1, args.num_epochs, train_loss, time.time() - start
        ))
        

        ##########################    valid    ##########################
        # 验证集数据管道（通常不 shuffle）
        validset = tf.data.TFRecordDataset(args.validset_tfrecords_path)
        validset = validset.map(map_function, num_parallel_calls=num_cpus)
        validset = validset.batch(batch_size=args.batch_size, drop_remainder=False)
        validset = validset.prefetch(buffer_size=args.batch_size)

        valid_loss_epoch = []
        valid_loss = 0.0
        start = time.time()

        for step, _input in enumerate(validset):
            loss_value = valid_step(_input, std)
            loss_float = float(loss_value.numpy() if hasattr(loss_value, "numpy") else loss_value)
            valid_loss_epoch.append(loss_float)
            valid_loss += loss_float

        valid_loss /= (step + 1)
        valid_loss_all.append(np.array(valid_loss_epoch, dtype=np.float32))

        print("valid epoch {}/{}, valid_loss = {:.06f}, time = {:.06f}".format(
            epoch + 1, args.num_epochs, valid_loss, time.time() - start
        ))
        print()

        ###################    save the train network    ###################
        # === 修改点：保存频率从 1000 调整为 10 ===
        if (epoch + 1) % 5 == 0:
            saved_model_dir = os.path.join(saved_model, f"{epoch + 1}_epochs")
            os.makedirs(saved_model_dir, exist_ok=True)  # 目录存在也不报错

            # 分别保存四个子网（结构+权重，.h5）
            sem_enc.save(os.path.join(saved_model_dir, "sem_enc.h5"))
            chan_enc.save(os.path.join(saved_model_dir, "chan_enc.h5"))
            chan_dec.save(os.path.join(saved_model_dir, "chan_dec.h5"))
            sem_dec.save(os.path.join(saved_model_dir, "sem_dec.h5"))

            # # 保存训练/验证损失（覆盖为最新）
            # if os.path.exists(train_loss_file):
            #     os.remove(train_loss_file)
            # sio.savemat(train_loss_file, {"train_loss": np.array(train_loss_all, dtype=np.float32)})

            # if os.path.exists(valid_loss_file):
            #     os.remove(valid_loss_file)
            # sio.savemat(valid_loss_file, {"valid_loss": np.array(valid_loss_all, dtype=np.float32)})

            print(f"[Checkpoint] Saved at epoch {epoch + 1} -> {saved_model_dir}")
    # === 训练结束：一次性归档到独立目录 ===
    run_dir = save_run_outputs(
        train_loss_epoch_list=train_loss_epoch_means,     # 建议优先保存“按 epoch 均值”的损失
        valid_loss_epoch_list=valid_loss_epoch_means,
        args_dict=vars(args),
        extra={
            # 如果你还想把“每个 epoch 的逐 step 曲线”也存起来，方便之后排查，可一并放入 extra
            "train_loss_per_step_per_epoch": np.array(train_loss_all, dtype=np.object_),
            "valid_loss_per_step_per_epoch": np.array(valid_loss_all, dtype=np.object_),
        }
    )
    print("All results saved to:", run_dir)

