# -*- coding: utf-8 -*-
"""第 4 项:音频输入(麦克风)体检 — 用 SDK 的 media 录音接口,不另开 sounddevice。

录 5 秒麦克风音频(GStreamer 自动选 "Reachy Mini Audio" 卡,16kHz/2 声道),
打印采样率、声道数、总 RMS 及左右声道各自 RMS,保存为 wav 供回放确认。
退出时上下文管理器自动释放音频资源。
"""

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import sys
import time
import wave
import numpy as np
from reachy_mini import ReachyMini

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
REC_SECONDS = 8.0
SILENCE_RMS = 0.002        # 低于此 RMS 视为基本没收到声音(全静音/全 0)
WALL_TIMEOUT = 12.0        # 累积采样的墙钟安全上限


def rms(x: np.ndarray) -> float:
    """均方根能量。"""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def save_wav_int16(path: str, data: np.ndarray, samplerate: int, channels: int) -> None:
    """把 float32 [-1,1] 立体声数据存成 16-bit PCM wav。"""
    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")          # 小端 int16,(N, ch) 交错
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm.tobytes())


def main() -> bool:
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== 第4项:音频输入(麦克风)体检 ===")
    ok = True

    with ReachyMini(connection_mode="localhost_only", media_backend="default") as mini:
        sr = mini.media.get_input_audio_samplerate()
        ch = mini.media.get_input_channels()
        print(f"采样率              : {sr} Hz")
        print(f"声道数              : {ch}  (Lite ReSpeaker 双麦 → 应为 2)")

        target_samples = int(sr * REC_SECONDS)
        chunks = []
        total = 0

        # 开始录音(连接已完成,管线先起来)
        mini.media.start_recording()
        time.sleep(0.5)   # 让 GStreamer 录音管线起来,丢弃启动瞬间

        # 连接 + 管线都就绪后,给一个实时倒计时,自跑时卡这个点开口最准
        print("\n连接与录音管线已就绪。", flush=True)
        for c in (3, 2, 1):
            print(f"   准备…… {c}", flush=True)
            time.sleep(1.0)

        # 倒计时期间积压的样本全部丢弃,确保从"开始说话"那刻起干净录起
        while mini.media.get_audio_sample() is not None:
            pass

        print("\n" + "=" * 44)
        print(f"   ▶ 现在开始说话!连续说『测试一二三』,说满 {int(REC_SECONDS)} 秒别停")
        print("=" * 44, flush=True)

        t0 = time.time()
        while total < target_samples and (time.time() - t0) < WALL_TIMEOUT:
            sample = mini.media.get_audio_sample()
            if sample is None:
                time.sleep(0.005)
                continue
            chunks.append(sample)
            total += sample.shape[0]

        mini.media.stop_recording()
        print("\n" + "=" * 44)
        print("   ■ 录音结束,可以停了")
        print("=" * 44, flush=True)

        if not chunks:
            print("⚠ 没录到任何音频样本,麦克风通路异常")
            print("=== 失败 ===")
            return False

        audio = np.concatenate(chunks, axis=0)        # (N, 2) float32
        n, nch = audio.shape[0], audio.shape[1]
        dur = n / sr

        overall_rms = rms(audio)
        peak = float(np.max(np.abs(audio)))
        print(f"\n实际录到样本数      : {n}  (~{dur:.2f} 秒, {nch} 声道)")

        # 显著打印 RMS
        print("\n" + "=" * 44)
        print(f"   录音 RMS 能量(总) = {overall_rms:.5f}")
        for c in range(nch):
            ch_rms = rms(audio[:, c])
            tag = "左(右?)声道" if c == 0 else "右(左?)声道"
            mark = "✓ 有信号" if ch_rms > SILENCE_RMS else "✗ 几乎无信号"
            print(f"   声道 {c} RMS = {ch_rms:.5f}   {mark}")
        print(f"   峰值幅度 = {peak:.4f}")
        print("=" * 44)

        # 保存 wav
        wav_path = os.path.join(OUT_DIR, "mic_recording.wav")
        save_wav_int16(wav_path, audio, sr, nch)
        print(f"\nwav 已保存          : {wav_path}")

        # 判定
        ch_rms_list = [rms(audio[:, c]) for c in range(nch)]
        if overall_rms <= SILENCE_RMS:
            print(f"⚠ 总 RMS 低于静音阈值 {SILENCE_RMS},可能没收到声音")
            ok = False
        if any(r <= SILENCE_RMS for r in ch_rms_list):
            print(f"⚠ 有声道 RMS 低于阈值,可能某只麦克风无信号")
            ok = False

    print("\n音频资源已通过上下文管理器释放。")
    print("=== 通过 ===" if ok else "=== 失败 ===")
    print("\n请回放 wav 确认能听清你说的“测试一二三”。")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
