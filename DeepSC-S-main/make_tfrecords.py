# -*- coding: utf-8 -*-  # 指定源码文件编码为 UTF-8，保证中文注释不乱码
"""
DeepSC-S 推理 Demo（目录版，不递归；无 soundfile 依赖；逐行中文注释）
- 输入：--input_dir 目录下的所有 .wav（不递归子目录）
- 读取：librosa.load(sr=8000, mono=True) 自动重采样并转单声道
- 切块：窗口长度与训练一致（wav_size = F*stride + frame - stride）
- 信道：Rayleigh + AWGN（SNR dB 可配置，默认 8 dB）
- 推理：语义编码器 -> 信道编码器 -> 信道 -> 信道解码器 -> 语义解码器
- 输出：<原名>_recon.wav 写入 --out_dir（8 kHz，PCM16）
- 权重：四路径/ckpt_dir/ckpt_root 自动解析（优先级：四路径 > ckpt_dir > ckpt_root）
"""

# ========================== 标准库与三方库导入 ==========================
import os                                  # 操作系统相关：路径拼接、列目录、判断文件/目录等
import re                                  # 正则表达式：用于匹配形如 "100_epochs" 的权重目录
import uuid                                # 生成短 UUID，避免中间 npz 文件重名覆盖
import argparse                            # 命令行参数解析
import numpy as np                         # 数值计算库
import tensorflow as tf                    # 深度学习主框架（2.x）
import librosa                             # 音频读取/重采样/转单声道
from scipy.io.wavfile import write as wavwrite  # 用 scipy 写 WAV（PCM16），替代 soundfile

# 从项目的 models.py 导入四个子网工厂与信道模型（与训练 main.py 完全一致）
from models import sem_enc_model, chan_enc_model, Chan_Model, chan_dec_model, sem_dec_model

# ========================== 基础工具函数 ==========================

def setup_gpu():
    """让 TensorFlow 的 GPU 显存按需申请，避免一次性占满（便于与其他进程共存）。"""
    gpus = tf.config.experimental.list_physical_devices("GPU")  # 列出可用 GPU
    for g in gpus:                                              # 遍历每张 GPU
        tf.config.experimental.set_memory_growth(g, True)       # 启用“按需分配显存”

def db_to_std(snr_db: float) -> float:
    """
    将 SNR(dB) 转为复高斯噪声标准差 std（与训练一致假设：E[|x|^2]≈1）。
    推导：SNR ≈ 1 / (2*std^2)  =>  std = sqrt(1/(2*SNR_linear)), 其中 SNR_linear = 10^(SNR_dB/10)
    """
    snr_linear = 10.0 ** (snr_db / 10.0)                         # dB -> 线性
    return float(np.sqrt(1.0 / (2.0 * snr_linear)))              # 返回噪声标准差

def load_wav_8k_mono(path: str, target_sr: int = 8000):
    """
    读取 wav，并重采样到 8 kHz、转单声道（与数据预处理/训练一致）。
    返回：(float32 波形 y, 目标采样率 target_sr, 原始采样率 orig_sr[尽力探测])。
    """
    try:
        orig_sr = librosa.get_samplerate(path)                    # 尝试探测原采样率
    except Exception:
        orig_sr = None                                            # 失败则置 None（后续以 target_sr 替代）
    y, _ = librosa.load(path, sr=target_sr, mono=True)            # 读取&重采样&转单声道
    return y.astype(np.float32), target_sr, (orig_sr if orig_sr else target_sr)  # 统一返回 float32

def compute_wav_size(sr: int, frame_size_s: float, stride_size_s: float, num_frame: int) -> int:
    """
    根据训练/sem_enc_model 的窗口公式计算“每窗采样点数”：
    wav_size = num_frame * stride_length + frame_length - stride_length
    其中 frame_length = sr*frame_size_s；stride_length = sr*stride_size_s。
    """
    frame_length  = int(sr * frame_size_s)                         # 例如 8000 * 0.016 = 128
    stride_length = int(sr * stride_size_s)                        # 例如 8000 * 0.016 = 128
    return num_frame * stride_length + frame_length - stride_length  # 默认参数下=128*128=16384

def chunk_waveform(x: np.ndarray, win_len: int):
    """
    将 1D 波形 x 切成等长窗口（长度为 win_len）：
    - 若 len(x) >= win_len：
        * 若正好整分（n % win_len == 0）：直接 reshape -> [N, win_len]；last_len = win_len
        * 若不能整分：在尾部“零填充”到下一个整窗，再 reshape；last_len = n % win_len
    - 若 len(x) < win_len：
        * 自身重复（tile）到至少 win_len，取前 win_len 作为唯一一窗；last_len = 原长 n
    返回：chunks[N, win_len]（float32），last_len（最后一窗真实有效长度）
    """
    n = int(x.shape[0])                                            # 原始长度
    if n == 0:                                                     # 空信号边界
        return np.zeros((1, win_len), dtype=np.float32), 0
    if n >= win_len:                                               # 长段
        rem = n % win_len
        if rem == 0:                                               # 正好整分
            chunks = np.reshape(x[:n], (n // win_len, win_len))    # 直接整形
            last_len = win_len                                     # 最后一窗有效长度=win_len
            return chunks.astype(np.float32), last_len
        else:                                                      # 不能整分
            pad = np.zeros(win_len - rem, dtype=np.float32)        # 尾部补零
            x2 = np.concatenate([x, pad], axis=0)                  # 拼接补零后波形
            chunks = np.reshape(x2, (x2.shape[0] // win_len, win_len))  # 直接整形
            last_len = rem                                         # 记录最后一窗真实长度
            return chunks.astype(np.float32), last_len
    else:                                                          # 短段（不足一窗）
        reps = int(np.ceil(win_len / n))                           # 需要重复的次数
        x2 = np.tile(x, reps)[:win_len]                            # 重复后截取前 win_len
        last_len = n                                               # 真实长度=原长
        return x2[np.newaxis, :].astype(np.float32), last_len

def stitch_chunks(chunks: np.ndarray, last_len: int, win_len: int):
    """
    将按窗推理得到的若干块拼回 1D 波形：
    - 先按顺序拼接并拉平；
    - 若最后一窗有“补零/重复”，按 last_len 从尾部裁掉多余的 (win_len-last_len)。
    """
    if len(chunks) == 0:                                           # 防御式：空输入
        return np.zeros((0,), np.float32)
    y = np.concatenate(chunks, axis=0).reshape(-1,)                # 拼接拉平
    trim = (win_len - last_len) if last_len < win_len else 0       # 需要裁掉的长度
    if trim > 0:
        y = y[:-trim]                                              # 从尾部裁掉
    return y.astype(np.float32)                                    # 返回 float32

def dump_npz(out_dir: str, base: str, step_tag: str, arrays: dict):
    """
    保存中间特征为压缩 npz：文件名附短 UUID，避免覆盖。
    - out_dir：输出目录
    - base：通常取输入 wav 的基本名
    - step_tag：如 b0001_sem / b0001_chanenc 等
    - arrays：要保存的 numpy 数组字典
    """
    os.makedirs(out_dir, exist_ok=True)                            # 确保目录存在
    short = str(uuid.uuid4())[:8]                                  # 8 位短 UUID
    path = os.path.join(out_dir, f"{base}_{step_tag}_{short}.npz") # 拼文件名
    np.savez_compressed(path, **arrays)                            # 压缩保存
    print(f"[DUMP] {step_tag} -> {path}")                          # 打印保存路径

# ========================== 权重定位（四路径/目录/自动） ==========================

def try_auto_ckpt(ckpt_root: str):
    """
    在 ckpt_root 下寻找形如 '<N>_epochs' 的子目录，返回 N 最大的路径；找不到返回 None。
    目的：当只提供根目录时，自动选择“训练轮数最高”的权重目录作为默认。
    """
    if not os.path.isdir(ckpt_root):                               # 根目录不存在
        return None
    cand = []                                                      # 候选列表 (N, path)
    for name in os.listdir(ckpt_root):                             # 遍历根目录
        m = re.match(r"(\d+)_epochs$", name)                       # 匹配 "100_epochs"
        if m:
            cand.append((int(m.group(1)), os.path.join(ckpt_root, name)))
    if not cand:                                                   # 无匹配
        return None
    cand.sort(key=lambda x: x[0], reverse=True)                    # 按 N 降序
    return cand[0][1]                                              # 返回最大 N 的目录

def resolve_weight_paths(args):
    """
    解析四个子网权重路径（优先级：显式四路径 > --ckpt_dir > --ckpt_root 自动）。
    - 返回：(sem_enc_path, chan_enc_path, chan_dec_path, sem_dec_path)
    """
    if args.sem_enc and args.chan_enc and args.chan_dec and args.sem_dec:  # 四路径都提供
        return args.sem_enc, args.chan_enc, args.chan_dec, args.sem_dec
    if args.ckpt_dir:                                                       # 给了固定目录
        d = args.ckpt_dir
        return (os.path.join(d, "sem_enc.h5"),
                os.path.join(d, "chan_enc.h5"),
                os.path.join(d, "chan_dec.h5"),
                os.path.join(d, "sem_dec.h5"))
    auto_dir = try_auto_ckpt(args.ckpt_root)                                # 自动从根目录挑选
    if auto_dir:
        return (os.path.join(auto_dir, "sem_enc.h5"),
                os.path.join(auto_dir, "chan_enc.h5"),
                os.path.join(auto_dir, "chan_dec.h5"),
                os.path.join(auto_dir, "sem_dec.h5"))
    # 三种方式都失败 -> 报错提示
    raise FileNotFoundError(
        "未能定位权重文件：请至少提供四个 .h5 路径，或提供 --ckpt_dir，"
        "或确保 --ckpt_root 下存在形如 '<N>_epochs' 的目录。"
    )

# ========================== 构建并加载模型（仅一次） ==========================

def build_models_and_load(sr: int, frame_size_s: float, stride_size_s: float, num_frame: int,
                          sem_enc_path: str, chan_enc_path: str, chan_dec_path: str, sem_dec_path: str):
    """
    构建四个子网与信道层，并一次性加载 .h5 权重。
    返回：(sem_enc, chan_enc, chan_layer, chan_dec, sem_dec, args_m, frame_length, stride_length, wav_size)
    """
    frame_length  = int(sr * frame_size_s)                           # 每帧点数，如 128
    stride_length = int(sr * stride_size_s)                          # 帧移点数，如 128
    wav_size      = num_frame * stride_length + frame_length - stride_length  # 每窗点数，如 16384

    # 与训练 main.py 的 args 结构保持一致（尤其是各模块通道配置），保证图一致
    args_m = argparse.Namespace(
        sr=sr,
        num_frame=num_frame,
        frame_size=frame_size_s,
        stride_size=stride_size_s,
        sem_enc_outdims=[32, 128, 128, 128, 128, 128, 128],         # 语义编码器通道配置（需与训练一致）
        chan_enc_filters=[128],                                      # 信道编码器卷积通道
        chan_dec_filters=[128],                                      # 信道解码器卷积通道
        sem_dec_outdims=[128, 128, 128, 128, 128, 128, 32],          # 语义解码器通道配置
    )

    # 构建四个子网 + 信道层（Rayleigh + AWGN）
    sem_enc = sem_enc_model(frame_length, stride_length, args_m)     # 语义编码器（内部含 norm+帧化）
    chan_enc = chan_enc_model(frame_length, args_m)                  # 信道编码器
    chan_layer = Chan_Model(name="Channel_Model")                    # 信道仿真层
    chan_dec = chan_dec_model(frame_length, args_m)                  # 信道解码器
    sem_dec = sem_dec_model(frame_length, stride_length, args_m)     # 语义解码器（内部含 deframe+denorm）

    # 加载四个子网的 .h5 权重
    print("[INFO] Loading weights:")
    print("  sem_enc :", sem_enc_path)
    print("  chan_enc:", chan_enc_path)
    print("  chan_dec:", chan_dec_path)
    print("  sem_dec :", sem_dec_path)
    sem_enc.load_weights(sem_enc_path)                               # 加载语义编码器权重
    chan_enc.load_weights(chan_enc_path)                             # 加载信道编码器权重
    chan_dec.load_weights(chan_dec_path)                             # 加载信道解码器权重
    sem_dec.load_weights(sem_dec_path)                               # 加载语义解码器权重
    print("[INFO] Weights loaded.")                                  # 提示完成

    # 返回模型与关键尺寸
    return sem_enc, chan_enc, chan_layer, chan_dec, sem_dec, args_m, frame_length, stride_length, wav_size

# ========================== 单文件推理（在已加载模型上） ==========================

def infer_one_file_with_models(
    wav_in: str,                             # 当前输入 wav 文件路径
    out_dir: str,                            # 输出目录
    models_tuple,                            # build_models_and_load 返回的元组
    snr_db: float = 8.0,                     # 信道 SNR（dB）
    sr: int = 8000,                          # 采样率（与训练一致）
    frame_size_s: float = 0.016,             # 帧长（秒）
    stride_size_s: float = 0.016,            # 帧移（秒）
    num_frame: int = 128,                    # 帧数
    batch_size: int = 32,                    # 批大小（以“窗口”为粒度）
    dump_intermediate: bool = False,         # 是否保存中间特征（调试用）
    dump_dir: str = "./demo_intermediate",   # 中间特征保存目录
    save_resampled: bool = False,            # 是否另存 8k 重采样音频
    resampled_dir: str = "./demo_resampled_8k",  # 8k 音频输出目录
):
    """
    用已构建/已加载权重的模型，对单个 wav 执行推理，写出 *_recon.wav。
    """
    (sem_enc, chan_enc, chan_layer, chan_dec, sem_dec,         # 解包模型与尺寸
     args_m, frame_length, stride_length, wav_size) = models_tuple

    x, _, orig_sr = load_wav_8k_mono(wav_in, target_sr=sr)     # 读取并重采样到 8k，x: float32
    base = os.path.splitext(os.path.basename(wav_in))[0]       # 提取不含扩展名的基本名

    if orig_sr != sr:                                          # 若原始采样率不是 8k，说明发生了重采样
        print(f"[INFO] {base}: resampled {orig_sr} Hz -> {sr} Hz (mono)")
        if save_resampled:                                     # 若要求另存重采样 8k 音频
            os.makedirs(resampled_dir, exist_ok=True)          # 确保目录存在
            out_8k = os.path.join(resampled_dir, f"{base}_8k.wav")  # 目标路径
            x16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)  # 量化到 PCM16
            wavwrite(out_8k, sr, x16)                          # 用 scipy 写 WAV
            print(f"[SAVE] resampled -> {out_8k}")
    else:
        print(f"[INFO] {base}: already {sr} Hz (mono)")        # 原本就是 8k 单声道

    chunks, last_len = chunk_waveform(x, win_len=wav_size)     # 切为 [N, wav_size]
    print(f"[INFO] {base} | samples={len(x)}, chunks={chunks.shape[0]}, win={wav_size}, last_len={last_len}")

    std_tf = tf.constant(db_to_std(snr_db), dtype=tf.float32)  # 固定的噪声 std（Tensor 常量）
    outs = []                                                  # 用于收集每窗重建

    for bi in range(0, chunks.shape[0], batch_size):           # 分批遍历窗口
        batch = chunks[bi:bi+batch_size]                       # 取本批窗口
        batch_tf = tf.convert_to_tensor(batch, dtype=tf.float32)  # 转 TF 张量

        feat_sem, batch_mean, batch_var = sem_enc(batch_tf)    # 语义编码（含 norm+帧化）
        feat_chanenc = chan_enc(feat_sem)                      # 信道编码（映射到传输符号）

        if dump_intermediate:                                  # 可选：保存中间特征便于调试
            btag = f"b{bi//batch_size:04d}"                    # 本批标签
            dump_npz(dump_dir, base, f"{btag}_sem",            # 保存语义特征与 norm 统计量
                     {"feat_sem": feat_sem.numpy(),
                      "batch_mean": batch_mean.numpy(),
                      "batch_var": batch_var.numpy()})
            dump_npz(dump_dir, base, f"{btag}_chanenc",        # 保存信道编码输出
                     {"feat_chanenc": feat_chanenc.numpy()})

        feat_chan = chan_layer(feat_chanenc, std_tf)           # 通过 Rayleigh+AWGN 信道
        feat_dec  = chan_dec(feat_chan)                        # 信道解码（恢复特征）
        recon     = sem_dec([feat_dec, batch_mean, batch_var]) # 语义解码（deframe+denorm 得到波形）

        outs.append(recon.numpy())                             # 收集 numpy 结果

    outs = np.concatenate(outs, axis=0)                        # [N, wav_size]
    y = stitch_chunks(outs, last_len=last_len, win_len=wav_size)  # 拼回整段并裁掉补零/重复
    y = np.clip(y, -1.0, 1.0)                                  # 安全裁剪，避免溢出

    os.makedirs(out_dir, exist_ok=True)                        # 确保输出目录存在
    out_wav = os.path.join(out_dir, f"{base}_recon.wav")       # 输出文件路径
    y16 = (y * 32767.0).astype(np.int16)                       # 转 PCM16
    wavwrite(out_wav, sr, y16)                                 # 用 scipy 写 WAV（不依赖 libsndfile）
    print(f"[DONE] {os.path.basename(wav_in)} -> {out_wav}")   # 打印完成信息
    return out_wav                                             # 返回输出路径

# ========================== CLI 参数解析 ==========================

def parse_args():
    """定义并解析本脚本的命令行参数。"""
    p = argparse.ArgumentParser(description="DeepSC-S 目录批量推理（不递归；无 soundfile 依赖）")
    # 输入/输出
    p.add_argument("--input_dir", required=True, help="包含 .wav 的文件夹（不递归）")         # 必填：输入目录
    p.add_argument("--out_dir", default="./demo_out", help="输出目录")                       # 选填：输出目录
    # 权重路径（优先级：四路径 > ckpt_dir > ckpt_root 自动）
    p.add_argument("--sem_enc", default="", help="sem_enc.h5")                               # 语义编码器权重
    p.add_argument("--chan_enc", default="", help="chan_enc.h5")                             # 信道编码器权重
    p.add_argument("--chan_dec", default="", help="chan_dec.h5")                             # 信道解码器权重
    p.add_argument("--sem_dec", default="", help="sem_dec.h5")                               # 语义解码器权重
    p.add_argument("--ckpt_dir", default="", help="如：/workspace/trained_outputs/saved_model/100_epochs")
    p.add_argument("--ckpt_root", default="/workspace/trained_outputs/saved_model",
                   help="未提供 --ckpt_dir/四路径时，从此目录自动选择最大 <N>_epochs 子目录")
    # 信道与帧参数（需与训练一致）
    p.add_argument("--snr_db", type=float, default=8.0, help="推理时通过信道的 SNR(dB)，建议与训练一致（8）")
    p.add_argument("--sr", type=int, default=8000, help="采样率（必须与训练一致）")
    p.add_argument("--frame_size", type=float, default=0.016, help="帧长（秒），默认 0.016")
    p.add_argument("--stride_size", type=float, default=0.016, help="帧移（秒），默认 0.016（无重叠）")
    p.add_argument("--num_frame", type=int, default=128, help="每窗包含的帧数，默认 128")
    p.add_argument("--batch_size", type=int, default=32, help="推理批大小（窗口级），默认 32")
    # 可选：保存中间特征 / 另存重采样后的 8k 音频
    p.add_argument("--dump_intermediate", action="store_true",
                   help="保存中间特征（语义编码输出、信道编码输出、batch_mean/var）")
    p.add_argument("--dump_dir", default="./demo_intermediate", help="中间特征保存目录")
    p.add_argument("--save_resampled", action="store_true",
                   help="若开启，则把重采样后的 8kHz 音频另存为 <原名>_8k.wav")
    p.add_argument("--resampled_dir", default="./demo_resampled_8k", help="保存 8k 音频的目录")
    return p.parse_args()                                           # 返回解析结果

# ========================== 程序入口 ==========================

if __name__ == "__main__":                                         # 仅脚本直接运行时执行
    setup_gpu()                                                     # 配置 GPU 显存按需分配
    args = parse_args()                                             # 解析命令行参数

    # 解析权重路径（四路径 > ckpt_dir > ckpt_root 自动）
    sem_enc_p, chan_enc_p, chan_dec_p, sem_dec_p = resolve_weight_paths(args)

    # 列出 input_dir 下所有 .wav（不递归）
    if not os.path.isdir(args.input_dir):                           # 目录存在性检查
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
    wavs = [
        os.path.join(args.input_dir, f)                             # 拼接完整路径
        for f in sorted(os.listdir(args.input_dir))                 # 按文件名排序，固定处理顺序
        if f.lower().endswith(".wav") and os.path.isfile(os.path.join(args.input_dir, f))  # 仅文件
    ]
    if not wavs:                                                    # 无 wav 则报错
        raise ValueError(f"目录下未找到 .wav 文件: {args.input_dir}")

    print(f"[INFO] Found {len(wavs)} wav(s) in {args.input_dir}")   # 打印统计

    # 一次性构建并加载模型（严格对齐训练窗口公式）
    models_tuple = build_models_and_load(
        sr=args.sr,
        frame_size_s=args.frame_size,
        stride_size_s=args.stride_size,
        num_frame=args.num_frame,
        sem_enc_path=sem_enc_p,
        chan_enc_path=chan_enc_p,
        chan_dec_path=chan_dec_p,
        sem_dec_path=sem_dec_p,
    )

    ok, fail = 0, 0                                                 # 成功/失败计数
    for w in wavs:                                                  # 遍历每个 wav
        try:
            infer_one_file_with_models(                             # 调用单文件推理
                wav_in=w,
                out_dir=args.out_dir,
                models_tuple=models_tuple,
                snr_db=args.snr_db,
                sr=args.sr,
                frame_size_s=args.frame_size,
                stride_size_s=args.stride_size,
                num_frame=args.num_frame,
                batch_size=args.batch_size,
                dump_intermediate=args.dump_intermediate,
                dump_dir=args.dump_dir,
                save_resampled=args.save_resampled,
                resampled_dir=args.resampled_dir,
            )
            ok += 1                                                 # 成功 +1
        except Exception as e:                                      # 捕获单文件异常但不中断
            print(f"[FAIL] {w}: {e}")                               # 打印失败原因
            fail += 1                                               # 失败 +1

    print(f"\n完成: 成功 {ok} 个, 失败 {fail} 个")                    # 汇总打印
