# -*- coding: utf-8 -*-
"""第 5 项:音频输出(扬声器)体检 — 用 SDK 的 media.play_sound() 播放。

1) 生成 440Hz 正弦波 wav 并播放,确认扬声器出声;
2) 用 pyttsx3(中文 Huihui 语音)合成一句中文存成 wav,再用 play_sound 播放。
play_sound 在 Windows 上经 get_audio_device("Sink") 选中 "Reachy Mini Audio" 声卡,
即从机器人自带扬声器出声(脚本会打印选中的 sink 作证据)。
"""

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import sys
import time
import wave
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
TONE_PATH = os.path.join(OUT_DIR, "test_tone_440hz.wav")
TTS_PATH = os.path.join(OUT_DIR, "tts_zh.wav")
TTS_TEXT = "你好,我是 Reachy Mini,音频测试正常"


def gen_sine_wav(path: str, freq: float = 440.0, seconds: float = 2.0,
                 sr: int = 44100, amp: float = 0.3) -> None:
    """生成单声道 16-bit 正弦波 wav。"""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wave_data = (amp * np.sin(2 * np.pi * freq * t))
    # 头尾 50ms 淡入淡出,避免爆音
    fade = int(sr * 0.05)
    env = np.ones_like(wave_data)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    pcm = np.clip(wave_data * env, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def gen_tts_wav(path: str, text: str) -> bool:
    """用 pyttsx3 选中文语音合成到 wav。返回是否成功选到中文语音。"""
    import pyttsx3
    engine = pyttsx3.init()
    zh = None
    for v in engine.getProperty("voices"):
        langs = str(getattr(v, "languages", [])).lower()
        if "zh" in langs or "chinese" in v.name.lower() or "ZH-CN" in v.id.upper():
            zh = v.id
            break
    if zh is not None:
        engine.setProperty("voice", zh)
    engine.setProperty("rate", 165)
    engine.save_to_file(text, path)
    engine.runAndWait()
    engine.stop()
    return zh is not None


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def main() -> bool:
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== 第5项:音频输出(扬声器)体检 ===")
    ok = True

    # --- 先离线生成两个 wav(不需要机器人)---
    print("生成 440Hz 正弦波 wav ...", flush=True)
    gen_sine_wav(TONE_PATH)
    print(f"  -> {TONE_PATH}  ({wav_duration(TONE_PATH):.1f}s)")

    print("用 pyttsx3 合成中文 TTS wav ...", flush=True)
    has_zh = gen_tts_wav(TTS_PATH, TTS_TEXT)
    if not os.path.exists(TTS_PATH) or os.path.getsize(TTS_PATH) < 1000:
        print("⚠ TTS wav 生成失败或为空")
        ok = False
    else:
        print(f"  -> {TTS_PATH}  ({wav_duration(TTS_PATH):.1f}s)  "
              f"中文语音:{'已选中 Huihui(zh-CN)' if has_zh else '⚠ 未找到中文语音,用了默认英文'}")

    # --- 连接机器人,确认输出声卡,依次播放 ---
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    from reachy_mini import ReachyMini
    from reachy_mini.media.device_detection import gst_monitor_devices, get_audio_device

    with ReachyMini(connection_mode="localhost_only", media_backend="default") as mini:
        # 打印选中的输出声卡(证据:声音从哪出)。须在 ReachyMini 之后,
        # 此时 GStreamer 已 Gst.init(),DeviceMonitor 才能用。
        print("\n输出声卡(Sink)枚举:")
        try:
            sinks = gst_monitor_devices("Audio/Sink")
            for d in sinks:
                mark = "  <== Reachy 扬声器" if "Reachy Mini Audio" in d.display_name else ""
                print(f"  - {d.display_name}{mark}")
            chosen = get_audio_device("Sink")
            print(f"SDK 实际选中的 Sink 设备 id:{chosen}")
        except Exception as e:
            print(f"  (枚举 Sink 出错,不影响播放:{e})")

        # 1) 播放正弦波
        print("\n" + "=" * 44)
        print("   ♪ 播放 440Hz 测试音(约 2 秒)—— 听机器人扬声器是否出声")
        print("=" * 44, flush=True)
        mini.media.play_sound(TONE_PATH)
        time.sleep(wav_duration(TONE_PATH) + 0.8)

        time.sleep(1.0)

        # 2) 播放中文 TTS
        print("\n" + "=" * 44)
        print(f"   🗣 播放中文 TTS:「{TTS_TEXT}」")
        print("=" * 44, flush=True)
        mini.media.play_sound(TTS_PATH)
        time.sleep(wav_duration(TTS_PATH) + 1.0)

    print("\n音频资源已通过上下文管理器释放。")
    print("=== 通过 ===" if ok else "=== 失败 ===")
    print("\n请确认:① 声音是否从机器人自带扬声器出来;② 那句中文是否听清。")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
