# -*- coding: utf-8 -*-
"""
批量重采样 WAV 文件到 8kHz
用法:
    python DeepSC-S-main/resample_to_8k.py --input_dir data_raw_16k --output_dir data_raw_8k
"""

import os
import argparse
import librosa
import soundfile as sf

def resample_wavs(input_dir: str, output_dir: str, target_sr: int = 8000):
    """批量重采样 WAV 文件"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    ok, fail = 0, 0
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".wav"):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        try:
            # 读取并重采样到 target_sr，强制单声道
            y, _ = librosa.load(in_path, sr=target_sr, mono=True)
            sf.write(out_path, y, target_sr)
            ok += 1
            print(f"[OK] {fname} -> {target_sr}Hz")
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")
            fail += 1

    print(f"\n完成: 成功 {ok} 个, 失败 {fail} 个")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重采样 WAV 文件到 8kHz")
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录 (16k wav 文件)")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录 (保存 8k wav 文件)")
    args = parser.parse_args()

    resample_wavs(args.input_dir, args.output_dir, target_sr=8000)
