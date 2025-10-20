# -*- coding: utf-8 -*-
"""
DeepSC-S 推理 Demo（目录版，不递归；无 soundfile/librosa 依赖）
- 输入：--input_dir 目录下的所有 .wav（不递归子目录）
- 读取：SciPy wavfile.read，必要时用 resample_poly 重采样到 8kHz，并转单声道
- 切块：窗口长度与训练一致（wav_size = F*stride + frame - stride）
- 信道：Rayleigh + AWGN（SNR dB 可配置，默认 8 dB）
- 推理：语义编码器 -> 信道编码器 -> 信道 -> 信道解码器 -> 语义解码器
- 输出：<原名>_recon.wav 写入 --out_dir（8 kHz，PCM16）
- 权重：四路径/ckpt_dir/ckpt_root 自动解析（优先级：四路径 > ckpt_dir > ckpt_root）
- 中间特征：--dump_intermediate 时保存 feat_sem、batch_mean/var、feat_chanenc 为 .npz
"""

import os
import re
import uuid
import argparse
import numpy as np
import tensorflow as tf

# 用 SciPy 完成 I/O（完全绕开 soundfile/librosa）
from scipy.io import wavfile as wavread      # 读 WAV
from scipy.signal import resample_poly       # 高质量有理数重采样
from scipy.io.wavfile import write as wavwrite  # 写 WAV（PCM16）

# 项目模型（与训练 main.py 一致）
from models import sem_enc_model, chan_enc_model, Chan_Model, chan_dec_model, sem_dec_model


# ----------------------- 基础工具 -----------------------

def setup_gpu():
    """按需申请显存，避免一次性占满。"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

def db_to_std(snr_db: float) -> float:
    """
    将 SNR(dB) 转为复高斯噪声标准差 std。
    假设 E[|x|^2]≈1，则 SNR ≈ 1/(2*std^2) => std = sqrt(1/(2*SNR_linear))
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(np.sqrt(1.0 / (2.0 * snr_linear)))

def load_wav_8k_mono(path: str, target_sr: int = 8000):
    """
    使用 SciPy 读取 WAV，转单声道并重采样到 target_sr（默认 8 kHz），完全避免 soundfile/librosa 依赖。
    返回：(float32 波形 y, 目标采样率 target_sr, 原始采样率 orig_sr)
    """
    sr, data = wavread.read(path)

    # 转单声道
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 归一化到 [-1, 1] 的 float32
    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float32) - 128.0) / 128.0
    else:
        y = data.astype(np.float32)
        maxv = float(np.max(np.abs(y))) if y.size else 1.0
        if maxv == 0.0:
            maxv = 1.0
        y = y / maxv

    # 必要时重采样到目标采样率
    if sr != target_sr:
        g = np.gcd(sr, target_sr)        # 最大公约数
        up = target_sr // g
        down = sr // g
        y = resample_poly(y, up, down).astype(np.float32)

    return y, target_sr, sr

def compute_wav_size(sr: int, frame_size_s: float, stride_size_s: float, num_frame: int) -> int:
    """
    与训练/sem_enc_model 一致的窗口长度：
    wav_size = num_frame * stride_length + frame_length - stride_length
    其中 frame_length = sr*frame_size_s；stride_length = sr*stride_size_s。
    """
    frame_length  = int(sr * frame_size_s)
    stride_length = int(sr * stride_size_s)
    return num_frame * stride_length + frame_length - stride_length

def chunk_waveform(x: np.ndarray, win_len: int):
    """
    切块逻辑与制 TFRecords 脚本对齐（强稳健）：
    - 若 len(x) > win_len：
        num_slices = n // win_len + 1
        x2 = concat(x, x)；x2 = x2[: win_len*num_slices]
        reshape -> [num_slices, win_len]
        last_len = n % win_len（若整除则为 win_len）
    - 若 len(x) <= win_len：
        反复自拼接到至少 win_len，取前 win_len，last_len=原长
    返回：chunks[N, win_len] (float32), last_len
    """
    n = int(x.shape[0])
    if n == 0:
        return np.zeros((1, win_len), np.float32), 0

    if n > win_len:
        num_slices = n // win_len + 1
        x2 = np.concatenate([x, x], axis=0)
        x2 = x2[: win_len * num_slices]
        chunks = np.reshape(x2, (num_slices, win_len))
        rem = n % win_len
        last_len = rem if rem > 0 else win_len
        return chunks.astype(np.float32), last_len
    else:
        x2 = x.copy()
        while x2.shape[0] < win_len:
            x2 = np.concatenate([x2, x2], axis=0)
        chunk = x2[:win_len]
        last_len = n
        return chunk[np.newaxis, :].astype(np.float32), last_len

def stitch_chunks(chunks: np.ndarray, last_len: int, win_len: int):
    """
    按顺序拼接并拉平；若最后一窗有补齐/自拼接，按 last_len 裁掉尾部多余 (win_len-last_len)。
    """
    if len(chunks) == 0:
        return np.zeros((0,), np.float32)
    y = np.concatenate(chunks, axis=0).reshape(-1,)
    trim = (win_len - last_len) if last_len < win_len else 0
    if trim > 0:
        y = y[:-trim]
    return y.astype(np.float32)

def dump_npz(out_dir: str, base: str, step_tag: str, arrays: dict):
    """将中间特征写为压缩 .npz，文件名附短 UUID 避免覆盖。"""
    os.makedirs(out_dir, exist_ok=True)
    short = str(uuid.uuid4())[:8]
    path = os.path.join(out_dir, f"{base}_{step_tag}_{short}.npz")
    np.savez_compressed(path, **arrays)
    print(f"[DUMP] {step_tag} -> {path}")


# ----------------------- 权重定位 -----------------------

def try_auto_ckpt(ckpt_root: str):
    """在 ckpt_root 下寻找 '<N>_epochs' 子目录，返回 N 最大的路径；找不到返回 None。"""
    if not os.path.isdir(ckpt_root):
        return None
    cand = []
    for name in os.listdir(ckpt_root):
        m = re.match(r"(\d+)_epochs$", name)
        if m:
            cand.append((int(m.group(1)), os.path.join(ckpt_root, name)))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1]

def resolve_weight_paths(args):
    """解析四个子网权重路径（优先级：显式四路径 > --ckpt_dir > --ckpt_root 自动）。"""
    if args.sem_enc and args.chan_enc and args.chan_dec and args.sem_dec:
        return args.sem_enc, args.chan_enc, args.chan_dec, args.sem_dec
    if args.ckpt_dir:
        d = args.ckpt_dir
        return (os.path.join(d, "sem_enc.h5"),
                os.path.join(d, "chan_enc.h5"),
                os.path.join(d, "chan_dec.h5"),
                os.path.join(d, "sem_dec.h5"))
    auto_dir = try_auto_ckpt(args.ckpt_root)
    if auto_dir:
        return (os.path.join(auto_dir, "sem_enc.h5"),
                os.path.join(auto_dir, "chan_enc.h5"),
                os.path.join(auto_dir, "chan_dec.h5"),
                os.path.join(auto_dir, "sem_dec.h5"))
    raise FileNotFoundError(
        "未能定位权重文件：请至少提供四个 .h5 路径，或提供 --ckpt_dir，"
        "或确保 --ckpt_root 下存在形如 '<N>_epochs' 的目录。"
    )


# ----------------------- 构建并加载模型（仅一次） -----------------------

def build_models_and_load(sr: int, frame_size_s: float, stride_size_s: float, num_frame: int,
                          sem_enc_path: str, chan_enc_path: str, chan_dec_path: str, sem_dec_path: str):
    """
    构建四个子网与信道层，并一次性加载 .h5 权重。
    返回：(sem_enc, chan_enc, chan_layer, chan_dec, sem_dec, args_m, frame_length, stride_length, wav_size)
    """
    frame_length  = int(sr * frame_size_s)
    stride_length = int(sr * stride_size_s)
    wav_size      = num_frame * stride_length + frame_length - stride_length

    # 与训练 main.py 的 args 结构保持一致
    args_m = argparse.Namespace(
        sr=sr,
        num_frame=num_frame,
        frame_size=frame_size_s,
        stride_size=stride_size_s,
        sem_enc_outdims=[32, 128, 128, 128, 128, 128, 128],
        chan_enc_filters=[128],
        chan_dec_filters=[128],
        sem_dec_outdims=[128, 128, 128, 128, 128, 128, 32],
    )

    sem_enc = sem_enc_model(frame_length, stride_length, args_m)
    chan_enc = chan_enc_model(frame_length, args_m)
    chan_layer = Chan_Model(name="Channel_Model")
    chan_dec = chan_dec_model(frame_length, args_m)
    sem_dec = sem_dec_model(frame_length, stride_length, args_m)

    print("[INFO] Loading weights:")
    print("  sem_enc :", sem_enc_path)
    print("  chan_enc:", chan_enc_path)
    print("  chan_dec:", chan_dec_path)
    print("  sem_dec :", sem_dec_path)
    sem_enc.load_weights(sem_enc_path)
    chan_enc.load_weights(chan_enc_path)
    chan_dec.load_weights(chan_dec_path)
    sem_dec.load_weights(sem_dec_path)
    print("[INFO] Weights loaded.")

    return sem_enc, chan_enc, chan_layer, chan_dec, sem_dec, args_m, frame_length, stride_length, wav_size


# ----------------------- 单文件推理 -----------------------

def infer_one_file_with_models(
    wav_in: str,
    out_dir: str,
    models_tuple,
    snr_db: float = 8.0,
    sr: int = 8000,
    frame_size_s: float = 0.016,
    stride_size_s: float = 0.016,
    num_frame: int = 128,
    batch_size: int = 32,
    dump_intermediate: bool = False,
    dump_dir: str = "./demo_intermediate",
    save_resampled: bool = False,
    resampled_dir: str = "./demo_resampled_8k",
):
    """用已加载好的模型对单个 wav 推理并写出 *_recon.wav。"""
    (sem_enc, chan_enc, chan_layer, chan_dec, sem_dec,
     args_m, frame_length, stride_length, wav_size) = models_tuple

    # 读取 & 重采样
    x, _, orig_sr = load_wav_8k_mono(wav_in, target_sr=sr)
    base = os.path.splitext(os.path.basename(wav_in))[0]

    if orig_sr != sr:
        print(f"[INFO] {base}: resampled {orig_sr} Hz -> {sr} Hz (mono)")
        if save_resampled:
            os.makedirs(resampled_dir, exist_ok=True)
            out_8k = os.path.join(resampled_dir, f"{base}_8k.wav")
            x16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
            wavwrite(out_8k, sr, x16)
            print(f"[SAVE] resampled -> {out_8k}")
    else:
        print(f"[INFO] {base}: already {sr} Hz (mono)")

    # 切块
    chunks, last_len = chunk_waveform(x, win_len=wav_size)
    print(f"[INFO] {base} | samples={len(x)}, chunks={chunks.shape[0]}, win={wav_size}, last_len={last_len}")

    std_tf = tf.constant(db_to_std(snr_db), dtype=tf.float32)
    outs = []

    # 批量前向
    for bi in range(0, chunks.shape[0], batch_size):
        batch = chunks[bi:bi+batch_size]
        batch_tf = tf.convert_to_tensor(batch, dtype=tf.float32)

        # 与 main.py 相同的前向顺序
        feat_sem, batch_mean, batch_var = sem_enc(batch_tf)
        feat_chanenc = chan_enc(feat_sem)

        if dump_intermediate:
            btag = f"b{bi//batch_size:04d}"
            dump_npz(dump_dir, base, f"{btag}_sem",
                     {"feat_sem": feat_sem.numpy(),
                      "batch_mean": batch_mean.numpy(),
                      "batch_var": batch_var.numpy()})
            dump_npz(dump_dir, base, f"{btag}_chanenc",
                     {"feat_chanenc": feat_chanenc.numpy()})

        feat_chan = chan_layer(feat_chanenc, std_tf)
        feat_dec  = chan_dec(feat_chan)
        recon     = sem_dec([feat_dec, batch_mean, batch_var])

        outs.append(recon.numpy())

    # 拼回整段并裁掉补齐
    outs = np.concatenate(outs, axis=0)
    y = stitch_chunks(outs, last_len=last_len, win_len=wav_size)
    y = np.clip(y, -1.0, 1.0)

    # 写出
    os.makedirs(out_dir, exist_ok=True)
    out_wav = os.path.join(out_dir, f"{base}_recon.wav")
    y16 = (y * 32767.0).astype(np.int16)
    wavwrite(out_wav, sr, y16)
    print(f"[DONE] {os.path.basename(wav_in)} -> {out_wav}")
    return out_wav


# ----------------------- CLI & main -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="DeepSC-S 目录批量推理（不递归；无 soundfile/librosa 依赖）")
    # 目录
    p.add_argument("--input_dir", required=True, help="包含 .wav 的文件夹（不递归）")
    p.add_argument("--out_dir", default="./demo_out", help="输出目录")
    # 权重
    p.add_argument("--sem_enc", default="", help="sem_enc.h5")
    p.add_argument("--chan_enc", default="", help="chan_enc.h5")
    p.add_argument("--chan_dec", default="", help="chan_dec.h5")
    p.add_argument("--sem_dec", default="", help="sem_dec.h5")
    p.add_argument("--ckpt_dir", default="", help="如：/workspace/trained_outputs/saved_model/100_epochs")
    p.add_argument("--ckpt_root", default="/workspace/trained_outputs/saved_model",
                   help="未提供 --ckpt_dir/四路径时，从此目录自动选择最大 <N>_epochs 子目录")
    # 帧与信道参数
    p.add_argument("--snr_db", type=float, default=8.0, help="推理时通过信道的 SNR(dB)")
    p.add_argument("--sr", type=int, default=8000, help="采样率（与训练一致）")
    p.add_argument("--frame_size", type=float, default=0.016, help="帧长（秒）")
    p.add_argument("--stride_size", type=float, default=0.016, help="帧移（秒）")
    p.add_argument("--num_frame", type=int, default=128, help="每窗包含的帧数")
    p.add_argument("--batch_size", type=int, default=32, help="推理批大小（窗口级）")
    # 中间特征 / 重采样原音频
    p.add_argument("--dump_intermediate", action="store_true",
                   help="保存中间特征（feat_sem、feat_chanenc、batch_mean/var）到 .npz")
    p.add_argument("--dump_dir", default="./demo_intermediate", help="中间特征保存目录")
    p.add_argument("--save_resampled", action="store_true",
                   help="若开启，则把重采样后的 8kHz 原音频另存为 <原名>_8k.wav")
    p.add_argument("--resampled_dir", default="./demo_resampled_8k", help="保存 8k 音频的目录")
    return p.parse_args()

if __name__ == "__main__":
    setup_gpu()
    args = parse_args()

    # 解析权重路径
    sem_enc_p, chan_enc_p, chan_dec_p, sem_dec_p = resolve_weight_paths(args)

    # 列出输入 wav（不递归）
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
    wavs = [
        os.path.join(args.input_dir, f)
        for f in sorted(os.listdir(args.input_dir))
        if f.lower().endswith(".wav") and os.path.isfile(os.path.join(args.input_dir, f))
    ]
    if not wavs:
        raise ValueError(f"目录下未找到 .wav 文件: {args.input_dir}")

    print(f"[INFO] Found {len(wavs)} wav(s) in {args.input_dir}")

    # 构建并加载模型
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

    # 批量推理
    ok, fail = 0, 0
    for w in wavs:
        try:
            infer_one_file_with_models(
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
            ok += 1
        except Exception as e:
            print(f"[FAIL] {w}: {e}")
            fail += 1

    print(f"\n完成: 成功 {ok} 个, 失败 {fail} 个")
