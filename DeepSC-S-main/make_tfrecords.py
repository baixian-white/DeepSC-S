# -*- coding: utf-8 -*-
"""
将一批 8kHz 单声道的 .wav 文件切成固定长度窗口（默认 16384 点 ≈ 2.048 秒），
做简单静音过滤后，序列化为 TFRecords，分别生成 train/valid 两个文件。

与论文《Semantic Communication Systems for Speech Transmission》电话系统设置对齐：
- W = 16384 = 128 帧 × 128 点/帧
- 采样率 sr = 8000
- frame_size = stride_size = 0.016 s
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import random
import timeit
import argparse
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

# 仅使用第 0 块 GPU；并按需申请显存，避免一次性占满
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ------------------------ 1) 命令行参数 ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Convert the set of .wavs to .TFRecords")

    # 采样率（脚本内部会强校验必须等于 8000）
    parser.add_argument("--sr", type=int, default=8000, help="sample rate for wav file")

    # 每个样本窗口由多少帧组成（论文里 F=128）
    parser.add_argument("--num_frame", type=int, default=128, help="number of frame in each batch")

    # 帧时长与帧移（秒）。默认相等 => 无重叠；如果想要重叠，可把 stride_size < frame_size
    parser.add_argument("--frame_size", type=float, default=0.016, help="time duration of each frame (seconds)")
    #注意，语音和视频不同，视频由图片组成，所以视频的帧时长受视频本身影响，但是语音的帧时长可以自己设定
    parser.add_argument("--stride_size", type=float, default=0.016, help="time duration of frame stride (seconds)")

    # 数据输入输出
    parser.add_argument("--wav_path", type=str, default="path of your original .wav files", help="path of wavset")
    parser.add_argument("--save_path", type=str, default="path to save .tfrecords files", help="path to save .tfrecords file")

    # 验证集比例 & 输出文件名
    parser.add_argument("--valid_percent", type=float, default=0.05, help="percent of validset in total dataset")
    parser.add_argument("--trainset_filename", type=str, default="trainset.tfrecords", help=".tfrecords filename of trainset")
    parser.add_argument("--validset_filename", type=str, default="validset.tfrecords", help=".tfrecords filename of validset")

    args = parser.parse_args()
    return args

args = parse_args()
print("Called with args:", args)

# ------------------------ 2) 推导窗口长度等全局参数 ------------------------
# 以采样点计的帧长 / 帧移
frame_length  = int(args.sr * args.frame_size)    # 8000 * 0.016 = 128
stride_length = int(args.sr * args.stride_size)   # 8000 * 0.016 = 128

# “一段样本”的总长度（单位：采样点）
# 公式由“滑动分帧”的基本关系推出：第一帧占 frame_length，其余 (num_frame-1) 帧每帧相隔 stride_length
# 化简后：num_frame*stride_length + frame_length - stride_length
window_size = args.num_frame * stride_length + frame_length - stride_length  # 128*128 = 16384（默认参数下）

# 批大小设定（若使用多 GPU，可把 num_gpu 改大，并保证训练时使用同样的 global_batch_size）
batch_size = 32
num_gpu = 1
global_batch_size = batch_size * num_gpu

# 输出文件扩展名校验
assert os.path.splitext(args.trainset_filename)[-1] == ".tfrecords", "extension of trainset_filename must be .tfrecords."
assert os.path.splitext(args.validset_filename)[-1] == ".tfrecords", "extension of validset_filename must be .tfrecords."

# ------------------------ 3) 主流程入口 ------------------------
if __name__ == "__main__":

    # --- TF Example 的 bytes 特征包装器 ---
    def bytes_feature(value: bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # --- 对单个 wav 文件执行：读取 -> 切片 -> 静音过滤 -> 写入 TFRecord ---
    def wav_processing(wav_file, tfrecords_file, window_size):
        # 读取 wav：sr 是采样率，wav_samples 是 int16 的一维数组
        sr, wav_samples = wavfile.read(wav_file)

        # 强制要求 8kHz（与论文电话系统实验对齐）
        if sr != 8000:
            raise ValueError("Sampling rate is expected to be 8kHz!")

        # 仅支持单通道数据
        assert wav_samples.ndim == 1, "check the size of wav_data (expect 1-D mono wav)."

        num_samples = wav_samples.shape[0]

        # 情形 A：音频长于一个窗口
        if num_samples > window_size:
            # +1 让后续 reshape 更稳健（随后会截断到窗口整数倍）
            num_slices = num_samples // window_size + 1

            # 将自身拼接一次，保证长度足够 reshape（避免最后一段残余不满一窗）
            wav_samples = np.concatenate((wav_samples, wav_samples), axis=0)
            wav_samples = wav_samples[0:window_size * num_slices]  # 截到整数窗口长度

            # 直接 reshape 成 [num_slices, window_size]，逐切片处理
            wav_slices = np.reshape(wav_samples, newshape=(num_slices, window_size))

            written = 0
            for wav_slice in wav_slices:
                # 以均值绝对幅度作为能量近似，阈值 0.015；小于则认为是“静音/低能量”，丢弃
                if np.mean(np.abs(wav_slice) / 2**15) < 0.015:
                    num_slices -= 1
                else:
                    wav_bytes = wav_slice.tobytes()   # 保留 int16 原始字节
                    example = tf.train.Example(
                        features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)})
                    )
                    tfrecords_file.write(example.SerializeToString())
                    written += 1

            return written  # 返回实际写入的切片数（已扣掉静音）

        # 情形 B：音频短于一个窗口
        else:
            num_slices = 1
            # 反复自拼接，直至长度至少一个窗口
            while wav_samples.shape[0] < window_size:
                wav_samples = np.concatenate((wav_samples, wav_samples), axis=0)

            # 截取前一个窗口长度
            wav_slice = wav_samples[0:window_size]

            # 静音过滤
            if np.mean(np.abs(wav_slice) / 2**15) < 0.015:
                num_slices -= 1  # 没写入
                return 0
            else:
                wav_bytes = wav_slice.tobytes()
                example = tf.train.Example(
                    features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)})
                )
                tfrecords_file.write(example.SerializeToString())
                return 1

    # ------------------------ 4) 列出 wav，划分训练/验证 ------------------------
    #从指定的 --wav_path 目录下，找到所有以 .wav 结尾的文件，拼接成完整路径，放到列表里。
    wav_files = [os.path.join(args.wav_path, f) for f in os.listdir(args.wav_path) if f.endswith(".wav")]
    #统计总共有多少个 .wav 文件
    num_wav_files = len(wav_files)
    #把列表 随机打乱顺序
    random.shuffle(wav_files)

    #按照参数 --valid_percent（默认 0.05）计算验证集的文件数
    num_validset_wav_files = int(args.valid_percent * num_wav_files)
    #剩下的全部归训练集
    num_trainset_wav_files = num_wav_files - num_validset_wav_files
    #形成验证集列表和训练集文件列表
    trainset_wav_files = wav_files[0:num_trainset_wav_files]
    validset_wav_files = wav_files[num_trainset_wav_files:num_wav_files]

    # 再次确认数量（可省）
    num_trainset_wav_files = len(trainset_wav_files)
    num_validset_wav_files = len(validset_wav_files)

    # 输出目录准备

    #确保 --save_path 存在，避免后面写 TFRecord 报路径不存在的错
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # ------------------------ 5) 处理训练集 ------------------------
    print("**********  Start processing and writing trainset  **********")
    #训练集 TFRecords 文件完整路径
    trainset_tfrecords_filepath = os.path.join(args.save_path, args.trainset_filename)
    #累计已写入的样本数（以“切片/窗口”为单位）
    total_trainset_slices = 0
    #记录开始时间
    begin_time = timeit.default_timer()
    #创建 TFRecordWriter 对象，准备写入
    trainset_tfrecords_file = tf.io.TFRecordWriter(trainset_tfrecords_filepath)

    # 遍历训练集 wav，逐文件切片并写入
    for file_count, trainset_wav_file in enumerate(trainset_wav_files):
        print("Processing trainset wav file {}/{} {}{}".format(
            file_count + 1, num_trainset_wav_files, trainset_wav_file, " " * 10
        ), end="\r")
        sys.stdout.flush()
        #调用前面定义的 wav_processing 函数，处理单个 wav 文件
        num_slices = wav_processing(trainset_wav_file, trainset_tfrecords_file, window_size)
        total_trainset_slices += num_slices

    # --- 批对齐：把样本总数补到 global_batch_size 的整数倍 ---
    print("**************   Post-processing trainset   **************")
    while total_trainset_slices % global_batch_size > 0:
        #从训练集的 wav 文件列表里随机挑一个文件，作为“补样本”的来源
        choose_wav_file = random.choice(trainset_wav_files)
        # 读取 wav
        sr, wav_samples = wavfile.read(choose_wav_file)
        #拿到这段音频的总采样点数，后面要根据长度决定能切出几个“整窗”。
        num_samples = wav_samples.shape[0]
        #只有当这条语音至少有一个完整窗口的长度（默认 window_size=16384）时，才参与补齐
        if num_samples >= window_size:
            # 直接以窗口步长切整窗（不使用静音过滤函数，逻辑内联到下面 if）
            #按窗口长度 window_size 逐段遍历这条语音（0→16384→32768→…），整窗步进
            for i in range(0, num_samples, window_size):
                if i + window_size > num_samples:
                    break
                wav_slice = wav_samples[i:i + window_size]
                #做一个能量阈值过滤：把 int16 归一化到 [-1,1] 后，取平均绝对值，要求 > 0.015 才认为是“非静音/有效片段”。与主流程的静音过滤是等价的
                if np.mean(np.abs(wav_slice) / 2**15) > 0.015:  # 注意：这里用的是 “> 0.015”
                    #把 int16 的窗口切片转成原始字节序列，准备写入 TFRecord。
                    wav_bytes = wav_slice.tobytes()
                    #构建一条 TF Example，只有一个字段 wav_raw，类型是 bytes（使用前面定义的 bytes_feature 封装）
                    example = tf.train.Example(
                        features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)})
                    )
                    #把这条样本序列化后，顺序追加写入训练集的 TFRecord 文件
                    trainset_tfrecords_file.write(example.SerializeToString())
                    #累计写入样本数 +1
                    total_trainset_slices += 1
                    #一旦样本总数凑够了“批大小的整数倍”，就停止补齐（break 掉当前的 for 循环）。
                    if total_trainset_slices % global_batch_size == 0:
                        break

    trainset_tfrecords_file.close()
    end_time = timeit.default_timer() - begin_time
    print("\n" + "*" * 50)
    print("Total processing and writing time (train): {} s".format(end_time))
    print()

    # ------------------------ 6) 处理验证集 ------------------------
    print("**********  Start processing and writing validset  **********")
    validset_tfrecords_filepath = os.path.join(args.save_path, args.validset_filename)

    total_validset_slices = 0
    begin_time = timeit.default_timer()
    validset_tfrecords_file = tf.io.TFRecordWriter(validset_tfrecords_filepath)

    for file_count, validset_wav_file in enumerate(validset_wav_files):
        print("Processing validset wav file {}/{} {}{}".format(
            file_count + 1, num_validset_wav_files, validset_wav_file, " " * 10
        ), end="\r")
        sys.stdout.flush()

        num_slices = wav_processing(validset_wav_file, validset_tfrecords_file, window_size)
        total_validset_slices += num_slices

    print("**************   Post-processing validset   **************")
    while total_validset_slices % global_batch_size > 0:
        choose_wav_file = random.choice(validset_wav_files)
        sr, wav_samples = wavfile.read(choose_wav_file)
        num_samples = wav_samples.shape[0]

        if num_samples >= window_size:
            for i in range(0, num_samples, window_size):
                if i + window_size > num_samples:
                    break
                wav_slice = wav_samples[i:i + window_size]
                if np.mean(np.abs(wav_slice) / 2**15) > 0.015:
                    wav_bytes = wav_slice.tobytes()
                    example = tf.train.Example(
                        features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)})
                    )
                    validset_tfrecords_file.write(example.SerializeToString())

                    total_validset_slices += 1
                    if total_validset_slices % global_batch_size == 0:
                        break

    validset_tfrecords_file.close()
    end_time = timeit.default_timer() - begin_time
    print("\n" + "*" * 50)
    print("Total processing and writing time (valid): {} s".format(end_time))
