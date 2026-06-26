# -*- coding: utf-8 -*-
"""VIS_DEBUG MJPEG HTTP 调试预览服务 + Conversation Dashboard。

纯只读：只访问 State 字段和全局缓冲区，不修改任何状态。
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

from voice.config import (
    DECIMATE, DOA_GATE_FRESH_S, GATE_DEG, INSTRUCTIONS, SNAP_DIR,
    PLAY_HAND_V_MAX, PLAY_SCORE_MIN, PLAY_SIZE_OFF,
)
from voice.state import (
    State, log,
    _vis_log_buf, _vis_log_seq,
    _conv_events, _conv_turns, _conv_seq,
    _feedback_notes, _feedback_seq,
)


# ── PIL 中文文字渲染(OpenCV putText 不支持中文) ──
_PIL_FONT = None
_PIL_FONT_SMALL = None

def _init_pil_fonts():
    global _PIL_FONT, _PIL_FONT_SMALL
    if _PIL_FONT is not None:
        return
    try:
        from PIL import ImageFont
        _windir = os.environ.get("WINDIR", r"C:\Windows")
        _candidates = [
            # Windows(微软雅黑 / 黑体 / 宋体)
            os.path.join(_windir, "Fonts", "msyh.ttc"),
            os.path.join(_windir, "Fonts", "simhei.ttf"),
            os.path.join(_windir, "Fonts", "simsun.ttc"),
            os.path.join(_windir, "Fonts", "Deng.ttf"),
            # macOS
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            # Linux
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for p in _candidates:
            if os.path.exists(p):
                _PIL_FONT = ImageFont.truetype(p, 16)
                _PIL_FONT_SMALL = ImageFont.truetype(p, 13)
                return
        _PIL_FONT = ImageFont.load_default()
        _PIL_FONT_SMALL = _PIL_FONT
    except Exception:
        _PIL_FONT = None
        _PIL_FONT_SMALL = None


def _put_cjk_text(bgr, text: str, pos: tuple, color=(255, 255, 255), font=None):
    """在 BGR numpy 图上绘制可能含中文的文字。fallback 到 cv2.putText。"""
    import cv2 as _cv2
    has_cjk = any(ord(c) > 127 for c in text)
    if not has_cjk or _PIL_FONT is None:
        _cv2.putText(bgr, text, pos, _cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
        return
    try:
        from PIL import Image, ImageDraw
        pil_img = Image.fromarray(_cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text(pos, text, font=font or _PIL_FONT, fill=(color[2], color[1], color[0]))
        bgr[:] = _cv2.cvtColor(np.array(pil_img), _cv2.COLOR_RGB2BGR)
    except Exception:
        _cv2.putText(bgr, text, pos, _cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)


def vis_debug_server(st: State, port: int, stop: threading.Event) -> None:
    """VIS_DEBUG=1 时启动 MJPEG HTTP 服务，浏览器打开 http://localhost:{port} 查看实时标注帧。
    画面 = 视觉子进程实际看到的降采样帧（DECIMATE×），叠加：
      蓝框=人脸(u,v,h)  绿框=有效手(score≥阈值)  黄框=低置信度手(可能误检)
      左上角=状态机/头部目标/face_locked  右上角=帧时间戳"""
    import cv2 as _cv2
    _init_pil_fonts()
    import http.server
    import socketserver

    def _build_frame() -> bytes:
        with st.lock:
            rgb = st.dbg_frame_small
            det = st.dbg_det
            state_name = st.state
            ty = st.track_yaw
            tp = st.track_pitch
            locked = st.face_locked
            hand_at = st.hand_at
            # DOA 字段
            doa_resid = st.doa_resid_stable
            doa_conf = st.doa_confident
            doa_at = st.doa_at
            body_yaw = st.body_yaw_deg
            gate_open = st.dbg_gate_open
            sw_active = st.dbg_switching
            sw_phase = st.dbg_switch_phase
            sw_target = st.dbg_switch_target
            speaking = time.monotonic() < st.playback_end_estimate + 0.1
            person_id = st.current_person_id
            person_name = st.current_person_name
            identity_injected = st.identity_injected

        if rgb is None:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            _cv2.putText(blank, "Waiting for frame...", (20, 180),
                        _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            _, jpg = _cv2.imencode(".jpg", blank)
            return jpg.tobytes()

        bgr = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]

        # ── 人脸框：优先 track_views(每框常驻身份+trackid),否则回退 all_faces/face ──
        _track_views = det.get("track_views") if det else None
        _all_faces = det.get("all_faces") if det else None
        _doa_sel = det.get("doa_selected_idx") if det else None
        if _track_views:
            for _v in _track_views:
                _b = _v.get("box")
                if not _b:
                    continue
                _ax0, _ay0, _ax1, _ay1 = int(_b[0]), int(_b[1]), int(_b[2]), int(_b[3])
                _is_sel = bool(_v.get("selected"))
                _is_conf = bool(_v.get("confirmed"))
                _bcolor = (255, 80, 0) if _is_sel else (150, 150, 150)  # 选中=蓝, 其余=灰
                _cv2.rectangle(bgr, (_ax0, _ay0), (_ax1, _ay1), _bcolor, 2 if _is_sel else 1)
                # 顶部标签:身份(未确认 Unknown-N / 已确认真名)+ trackid —— 每框常驻
                _nm = _v.get("name") or "?"
                _top = f"{_nm} T{_v.get('track_id')}"
                _cjk_w = sum(18 if ord(c) > 127 else 10 for c in _top) + 6
                _lblc = (255, 80, 0) if _is_sel else ((0, 140, 0) if _is_conf else (90, 90, 90))
                _ly = _ay0 - 20 if _ay0 - 20 >= 0 else _ay1   # 靠顶则画到框下方
                _cv2.rectangle(bgr, (_ax0, _ly), (_ax0 + _cjk_w, _ly + 20), _lblc, -1)
                _put_cjk_text(bgr, _top, (_ax0 + 2, _ly + 2), (255, 255, 255))
        elif _all_faces and len(_all_faces) > 1:
            for _fi, _af in enumerate(_all_faces):
                _is_sel = (_doa_sel is not None and _fi == _doa_sel)
                _afu, _afv, _afh = _af["u"], _af["v"], _af["h"]
                _abox = _af.get("box")
                if _abox:   # 真实检测框(降采样像素,与画布同坐标系)→ 贴合
                    _ax0, _ay0 = int(_abox[0]), int(_abox[1])
                    _ax1, _ay1 = int(_abox[0] + _abox[2]), int(_abox[1] + _abox[3])
                else:       # 兜底:无 box 时用 u/v/h 估
                    _afw = _afh * 0.85
                    _ax0 = int((_afu - _afw / 2) * W)
                    _ay0 = int((_afv - _afh / 2) * H)
                    _ax1 = int((_afu + _afw / 2) * W)
                    _ay1 = int((_afv + _afh / 2) * H)
                _color = (255, 80, 0) if _is_sel else (120, 120, 120)
                _thick = 2 if _is_sel else 1
                _cv2.rectangle(bgr, (_ax0, _ay0), (_ax1, _ay1), _color, _thick)
                _ftag = f"DOA" if _is_sel else f"#{_fi}"
                _flbl = f"{_ftag} u={_afu:.2f} h={_afh:.2f}"
                _cv2.rectangle(bgr, (_ax0, _ay0 - 18), (_ax0 + len(_flbl) * 9, _ay0), _color, -1)
                _cv2.putText(bgr, _flbl, (_ax0 + 2, _ay0 - 4),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                if _is_sel and (person_name or person_id):
                    id_label = person_name or (person_id[:12] if person_id else "")
                    mem_s = "MEM" if identity_injected else ""
                    id_str = f"{id_label} {mem_s}".strip()
                    _cjk_w = sum(18 if ord(c) > 127 else 10 for c in id_str) + 4
                    _cv2.rectangle(bgr, (_ax0, _ay1), (_ax0 + _cjk_w, _ay1 + 22), (200, 60, 0), -1)
                    _put_cjk_text(bgr, id_str, (_ax0 + 2, _ay1 + 2), (255, 255, 255))
        elif det and det.get("face") is not None:
            fu, fv, fh = det["face"]
            _fbox = det.get("face_box")
            if _fbox:   # 真实检测框(降采样像素)→ 贴合
                fx0, fy0 = int(_fbox[0]), int(_fbox[1])
                fx1, fy1 = int(_fbox[0] + _fbox[2]), int(_fbox[1] + _fbox[3])
            else:       # 兜底:无 box 时用 u/v/h 估
                fw = fh * 0.85
                fx0 = int((fu - fw / 2) * W)
                fy0 = int((fv - fh / 2) * H)
                fx1 = int((fu + fw / 2) * W)
                fy1 = int((fv + fh / 2) * H)
            _cv2.rectangle(bgr, (fx0, fy0), (fx1, fy1), (255, 80, 0), 2)
            label = f"FACE u={fu:.2f} v={fv:.2f} h={fh:.2f} n={det.get('n_faces',1)}"
            _cv2.rectangle(bgr, (fx0, fy0 - 18), (fx0 + len(label) * 9, fy0), (255, 80, 0), -1)
            _cv2.putText(bgr, label, (fx0 + 2, fy0 - 4),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if person_name or person_id:
                id_label = person_name or (person_id[:12] if person_id else "")
                mem_s = "MEM" if identity_injected else ""
                id_str = f"{id_label} {mem_s}".strip()
                _cjk_w = sum(18 if ord(c) > 127 else 10 for c in id_str) + 4
                _cv2.rectangle(bgr, (fx0, fy1), (fx0 + _cjk_w, fy1 + 22), (200, 60, 0), -1)
                _put_cjk_text(bgr, id_str, (fx0 + 2, fy1 + 2), (255, 255, 255))

        # ── 手部框（绿=有效 / 黄=低置信 / 橙=底部过滤）──
        if det and det.get("hand") is not None:
            h = det["hand"]
            hu, hv, hsize = h.get("u", 0.5), h.get("v", 0.5), h.get("size", 0.0)
            hscore = h.get("score", 0.0)
            # bbox: hsize is the max(dx,dy) in normalised coords — apply to each axis separately
            half_w = int(hsize * W / 2)
            half_h = int(hsize * H / 2)
            hx0 = max(0, int(hu * W) - half_w)
            hy0 = max(0, int(hv * H) - half_h)
            hx1 = min(W - 1, int(hu * W) + half_w)
            hy1 = min(H - 1, int(hv * H) + half_h)
            valid = hscore >= PLAY_SCORE_MIN and hsize >= PLAY_SIZE_OFF and hv <= PLAY_HAND_V_MAX
            color = (0, 200, 0) if valid else ((0, 120, 255) if hv > PLAY_HAND_V_MAX else (0, 200, 255))  # 绿/橙(底部)/黄
            tag = "HAND" if valid else ("HAND(BOT)" if hv > PLAY_HAND_V_MAX else "HAND(LOW)")
            _cv2.rectangle(bgr, (hx0, hy0), (hx1, hy1), color, 2)
            fingers = h.get("fingers", -1)
            gesture = h.get("gesture") or ""
            g_str = f" [{gesture}]" if gesture else (f" {fingers}f" if fingers >= 0 else "")
            hlabel = f"{tag} sz={hsize:.2f} sc={hscore:.2f} v={hv:.2f}{g_str}"
            lbl_y = min(hy1 + 18, H - 4)
            _cv2.rectangle(bgr, (hx0, lbl_y - 16), (hx0 + len(hlabel) * 9, lbl_y + 2), color, -1)
            _cv2.putText(bgr, hlabel, (hx0 + 2, lbl_y - 2),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            # v-threshold line
            vy = int(PLAY_HAND_V_MAX * H)
            _cv2.line(bgr, (0, vy), (W, vy), (0, 120, 255), 1)
            _cv2.putText(bgr, f"v_max={PLAY_HAND_V_MAX}", (4, vy - 4),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 120, 255), 1)

        # ── 左上角：状态机信息（白字黑底）──
        _tnow = time.time()
        now_s = time.strftime("%H:%M:%S", time.localtime(_tnow)) + f".{int((_tnow % 1) * 1000):03d}"
        lines = [
            f"[{state_name}]",
            f"yaw={ty:+.1f}deg  pitch={tp:+.1f}deg",
            f"face_locked={'Y' if locked else 'N'}  hand_age={time.monotonic()-hand_at:.1f}s",
            now_s,
        ]
        for i, line in enumerate(lines):
            y = 18 + i * 20
            _cv2.rectangle(bgr, (0, y - 15), (len(line) * 9 + 4, y + 4), (0, 0, 0), -1)
            _cv2.putText(bgr, line, (2, y),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                         (0, 255, 255) if i == 0 else (255, 255, 255), 1)

        # ── 右上角：醒目大时间戳（HH:MM:SS.mmm,与 log 对应,便于定位问题帧）──
        (_tsw, _tsh), _ = _cv2.getTextSize(now_s, _cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        _cv2.rectangle(bgr, (W - _tsw - 12, 2), (W, _tsh + 14), (0, 0, 0), -1)
        _cv2.putText(bgr, now_s, (W - _tsw - 8, _tsh + 8),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── 底部诊断行：区分"没检出"和"管线未写入"──
        if det is None:
            diag = "det=None  vision_result_loop not writing (crashed?)"
            diag_color = (0, 0, 255)   # 红
        else:
            face_s = f"face={det['face']}" if det.get("face") else "face=None"
            hand_s = f"hand=size{det['hand']['size']:.2f} score{det['hand']['score']:.2f}" if det.get("hand") else "hand=None"
            diag = f"{face_s}  {hand_s}  n={det.get('n_faces',0)}"
            diag_color = (0, 255, 0) if det.get("face") or det.get("hand") else (100, 100, 100)
        _cv2.rectangle(bgr, (0, H - 22), (W, H), (0, 0, 0), -1)
        _cv2.putText(bgr, diag, (4, H - 6),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.42, diag_color, 1)

        # ── DOA 弧条（底部第二行，高 38px）──
        _doa_h = 38          # DOA 条高度
        _doa_y0 = H - 22 - _doa_h  # 在底部诊断行上方
        _doa_mx = 12         # 左右边距
        _doa_range = 90.0    # ±90°
        _gate_deg = GATE_DEG  # ±55° 门控范围

        def _deg2x(deg: float) -> int:
            return int(_doa_mx + (deg + _doa_range) / (2 * _doa_range) * (W - 2 * _doa_mx))

        # 背景黑条
        _cv2.rectangle(bgr, (0, _doa_y0), (W, H - 22), (20, 20, 20), -1)

        # 门控范围背景：±GATE_DEG 内绿透明叠，外红透明叠
        _gate_x0 = _deg2x(-_gate_deg)
        _gate_x1 = _deg2x(_gate_deg)
        _overlay = bgr.copy()
        _cv2.rectangle(_overlay, (_doa_mx, _doa_y0 + 2), (_gate_x0, H - 24), (0, 0, 80), -1)   # 左侧超范围=红
        _cv2.rectangle(_overlay, (_gate_x1, _doa_y0 + 2), (W - _doa_mx, H - 24), (0, 0, 80), -1)  # 右侧超范围=红
        _cv2.rectangle(_overlay, (_gate_x0, _doa_y0 + 2), (_gate_x1, H - 24), (0, 60, 0), -1)  # 范围内=绿
        _cv2.addWeighted(_overlay, 0.4, bgr, 0.6, 0, bgr)

        # 刻度线：画面左=机器人左(resid+)，画面右=机器人右(resid-)
        for _d in (-90, -60, -30, 0, 30, 60, 90):
            _tx = _deg2x(float(-_d))
            _cv2.line(bgr, (_tx, _doa_y0 + 2), (_tx, _doa_y0 + 8), (120, 120, 120), 1)
            if _d != 0:
                _lbl = f"{_d:+d}"
                _cv2.putText(bgr, _lbl, (_tx - 10, _doa_y0 + 20),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)
            else:
                _cv2.line(bgr, (_tx, _doa_y0 + 2), (_tx, _doa_y0 + 14), (180, 180, 180), 1)

        # body_yaw 三角标（白色，朝下）
        _bx = _deg2x(float(np.clip(-body_yaw, -_doa_range, _doa_range)))
        _tri = np.array([[_bx, _doa_y0 + 2], [_bx - 5, _doa_y0 + 10], [_bx + 5, _doa_y0 + 10]], np.int32)
        _cv2.fillPoly(bgr, [_tri], (220, 220, 220))
        _cv2.putText(bgr, "H", (_bx - 4, _doa_y0 + 10),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)

        # 切换目标角（橙色三角，朝上，切换中才显示）
        if sw_active:
            _sx = _deg2x(float(np.clip(-sw_target, -_doa_range, _doa_range)))
            _stri = np.array([[_sx, H - 25], [_sx - 5, H - 33], [_sx + 5, H - 33]], np.int32)
            _cv2.fillPoly(bgr, [_stri], (0, 130, 255))  # 橙

        # DOA 方向箭头（主指示器，从中央向外）
        _doa_fresh = doa_resid is not None and (time.monotonic() - doa_at) < DOA_GATE_FRESH_S
        if doa_resid is not None:
            _dx = _deg2x(float(np.clip(-doa_resid, -_doa_range, _doa_range)))
            _cy_bar = (_doa_y0 + H - 22) // 2
            if doa_conf and _doa_fresh:
                _arrow_c = (0, 220, 0)   # 绿：confident + fresh
            elif _doa_fresh:
                _arrow_c = (0, 180, 255)  # 橙：fresh 但不 confident
            else:
                _arrow_c = (80, 80, 80)   # 灰：stale
            _cv2.arrowedLine(bgr, (_deg2x(0.0), _cy_bar), (_dx, _cy_bar),
                             _arrow_c, 2, tipLength=0.2)
            _cv2.circle(bgr, (_dx, _cy_bar), 4, _arrow_c, -1)

        # 右下角 DOA 文字状态
        _now_m = time.monotonic()
        _fresh_s = f"{_now_m - doa_at:.1f}s" if doa_resid is not None else "—"
        _resid_s = f"{doa_resid:+.0f}°" if doa_resid is not None else "—"
        _gate_s = "OPEN" if gate_open else "BLOCK"
        _gate_c = (0, 220, 0) if gate_open else (0, 0, 220)
        _spk_s = "SPK" if speaking else ""
        _sw_s = f"SW:{sw_phase}" if sw_active else ""
        _conf_s = "conf" if doa_conf else "unc"
        _doa_line = f"DOA {_resid_s} {_conf_s} {_fresh_s}  gate:{_gate_s}  {_sw_s}  {_spk_s}"
        _txt_w = len(_doa_line) * 8 + 4
        _cv2.rectangle(bgr, (W - _txt_w - 2, _doa_y0 + 2), (W - 2, _doa_y0 + 18), (0, 0, 0), -1)
        _cv2.putText(bgr, _doa_line, (W - _txt_w, _doa_y0 + 14),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.38, _gate_c, 1)

        _, jpg = _cv2.imencode(".jpg", bgr, [_cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpg.tobytes()

    _VIS_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>小艺 Debug</title>
<style>
:root{--bg:#09090f;--card:#131320;--bdr:#1e1e30;--txt:#dde1ea;--muted:#505877;
     --green:#22d3a0;--red:#f25e6b;--orange:#f5a623;--blue:#38bdf8;--purple:#a78bfa;
     --mono:'SF Mono','Fira Code',Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--txt);font:13px/1.4 system-ui,sans-serif;overflow:hidden}
#hdr{display:flex;align-items:center;gap:10px;padding:5px 14px;background:var(--card);
     border-bottom:1px solid var(--bdr);height:40px;flex-shrink:0}
#hdr h1{font-size:14px;font-weight:600;letter-spacing:-.2px}
.badge{padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.ba{background:#1a1a2e;color:var(--muted)}.be{background:#3d2000;color:#fbbf24}
.bt{background:#003d28;color:var(--green)}.bs{background:#001d3d;color:var(--blue)}
.bp{background:#2a0057;color:var(--purple)}.br{background:#3d0010;color:var(--red)}
/* tab bar */
#tabs{display:flex;gap:2px;margin-left:16px}
.tab{padding:4px 14px;border-radius:6px;font-size:12px;font-weight:500;cursor:pointer;color:var(--muted);background:transparent;border:1px solid transparent;transition:.15s}
.tab.active{background:var(--card);border-color:var(--bdr);color:var(--txt)}
.tab:hover:not(.active){color:var(--txt)}
#dot{margin-left:auto;width:7px;height:7px;border-radius:50%;background:var(--muted);transition:background .4s}
#dot.ok{background:var(--green)}
/* views */
#view-camera,#view-conv{height:calc(100vh - 40px);display:none}
#view-camera.active,#view-conv.active{display:grid}
/* ── Camera view ── */
#view-camera{grid-template-columns:1fr 304px;grid-template-rows:1fr 200px;gap:5px;padding:5px}
#cv{background:#000;border-radius:8px;overflow:hidden;position:relative;display:flex;align-items:center;justify-content:center}
#cv img{max-width:100%;max-height:100%;object-fit:contain;display:block}
#cv-lbl{position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,.65);padding:2px 8px;border-radius:4px;font:10px var(--mono);color:var(--muted)}
#side{display:flex;flex-direction:column;gap:5px}
.card{background:var(--card);border:1px solid var(--bdr);border-radius:8px;padding:10px}
.ch{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin-bottom:8px}
#rc{flex:1;display:flex;flex-direction:column;min-height:0;overflow:hidden}
canvas{display:block;margin:0 auto}
.sr{display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid var(--bdr);gap:8px}
.sr:last-child{border:none}
.sl{color:var(--muted);white-space:nowrap;font-size:12px}
.sv{font:11px var(--mono);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#lw{grid-column:1/3;background:var(--card);border:1px solid var(--bdr);border-radius:8px;display:flex;flex-direction:column;overflow:hidden;position:relative}
#lh{padding:5px 12px;border-bottom:1px solid var(--bdr);flex-shrink:0;display:flex;align-items:center;gap:8px}
#lh span{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.7px;color:var(--muted)}
#lb{flex:1;overflow-y:auto;padding:4px 12px;font:11px/1.75 var(--mono);min-height:0}
#log-bottom-btn{display:none;position:absolute;right:16px;bottom:12px;width:28px;height:28px;border-radius:50%;background:#3b82f6;color:#fff;border:none;cursor:pointer;font-size:14px;line-height:28px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.3);z-index:10;opacity:.85;transition:opacity .15s}
#log-bottom-btn:hover{opacity:1}
.ll{white-space:pre-wrap;word-break:break-all}
.lk{color:var(--green)}.lw2{color:var(--orange)}.le{color:var(--red)}.ld{color:#6366f1}.lm{color:#4b5563}
/* ── Conversation view ── */
#view-conv{grid-template-columns:320px 1fr;grid-template-rows:1fr 180px 44px;gap:5px;padding:5px}
#turn-list{overflow-y:auto;display:flex;flex-direction:column;gap:4px;padding-right:2px}
.turn-card{background:var(--card);border:1px solid var(--bdr);border-radius:8px;padding:8px 10px;cursor:pointer;transition:border-color .15s,background .15s;height:auto;overflow:visible}
.turn-card:hover{border-color:var(--blue)}
.turn-card.selected{border-color:var(--blue);background:#0d1a26}
.tc-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.tc-id{font:10px var(--mono);color:var(--blue);font-weight:700}
.tc-ts{font:10px var(--mono);color:var(--muted)}
.tc-row{font-size:11px;color:var(--muted);padding:1px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tc-row .em{margin-right:4px}
.tc-asr{color:var(--txt)}
.tc-out{color:var(--green)}
.tc-tool{color:var(--orange)}
.tc-vlm{color:var(--purple)}
.tc-fb{color:var(--blue);font:10px var(--mono)}
.tc-ctx{font:10px var(--mono);color:#6366f1;opacity:.7;padding:2px 0 3px;border-bottom:1px solid #1a1a2e;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#event-panel{background:var(--card);border:1px solid var(--bdr);border-radius:8px;display:flex;flex-direction:column;overflow:hidden}
#ep-hdr{padding:5px 12px;border-bottom:1px solid var(--bdr);flex-shrink:0;font:10px var(--mono);color:var(--muted);display:flex;gap:12px;align-items:center}
#ep-list{flex:1;overflow-y:auto;font:11px var(--mono)}
.ev{display:grid;grid-template-columns:36px 84px 200px 1fr;gap:0 8px;padding:3px 10px;border-bottom:1px solid #0d0d18;cursor:pointer;align-items:center}
.ev:hover{background:#0d1020}
.ev.hl{background:#0d1a26;border-left:2px solid var(--blue)}
.ev .es{color:var(--muted);font-size:10px}
.ev .ets{color:var(--muted);font-size:10px}
.ev .ety{color:#6366f1;font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ev .elb{color:var(--txt);font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ev.role-user .elb{color:var(--blue)}
.ev.role-model .elb{color:var(--green)}
.ev.role-tool .elb{color:var(--orange)}
.ev.role-system .elb{color:var(--muted)}
/* Issue#2: highlight overlay for filtered events */
.ev.ev-hl{background:#0c1820;border-left:3px solid var(--blue)}
.ev.ev-hl .ety{color:#818cf8}
.ev.ev-dim{opacity:.28}
/* timeline canvas */
#timeline-wrap{grid-column:1/3;background:var(--card);border:1px solid var(--bdr);border-radius:8px;overflow:hidden;position:relative;display:flex;flex-direction:column}
#tl-lanes{position:absolute;left:0;top:0;bottom:0;width:44px;background:var(--card);z-index:2;border-right:1px solid var(--bdr);pointer-events:none}
#tl-scroll{flex:1;overflow-x:auto;overflow-y:hidden;position:relative;padding-left:44px}
#tl-latest-btn{position:absolute;right:10px;bottom:6px;background:#1e3a5f;border:1px solid #38bdf8;color:#38bdf8;font:11px system-ui;padding:3px 10px;border-radius:12px;cursor:pointer;z-index:10;display:none;opacity:.9}
#tl-latest-btn:hover{opacity:1}
#tl-tip{position:fixed;background:#1e1e2e;border:1px solid #374151;color:#e5e7eb;font:11px monospace;padding:4px 8px;border-radius:6px;pointer-events:none;z-index:200;display:none;max-width:280px;white-space:pre-wrap;line-height:1.4}
#timeline{display:block;height:100%}
/* feedback bar */
#fb-bar{grid-column:1/3;background:var(--card);border:1px solid var(--bdr);border-radius:8px;
        display:flex;align-items:center;gap:10px;padding:0 14px}
#fb-btn{padding:4px 16px;border-radius:20px;border:1px solid var(--bdr);background:#1a1a2e;color:var(--muted);
        font-size:12px;cursor:pointer;user-select:none;transition:.15s}
#fb-btn.recording{background:#3d0010;color:var(--red);border-color:var(--red)}
#fb-status{flex:1;font:11px var(--mono);color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* modal */
#payload-modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;align-items:center;justify-content:center}
#payload-modal.open{display:flex}
#payload-box{background:var(--card);border:1px solid var(--bdr);border-radius:10px;max-width:700px;width:90%;max-height:80vh;display:flex;flex-direction:column}
#pb-hdr{padding:10px 16px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:10px}
#pb-hdr h3{flex:1;font-size:13px}
#pb-close{background:none;border:none;color:var(--muted);font-size:18px;cursor:pointer;line-height:1}
#pb-body{flex:1;overflow-y:auto;padding:12px 16px;font:11px/1.7 var(--mono);color:var(--txt);white-space:pre-wrap;word-break:break-all}
</style></head>
<body>
<div id="hdr">
  <h1>&#x1F916; 小艺 Reachy Mini &mdash; Debug Dashboard</h1>
  <div id="tabs">
    <div class="tab active" onclick="switchTab('camera')">Camera</div>
    <div class="tab" onclick="switchTab('conv')">Conversation</div>
  </div>
  <span id="badge" class="badge ba">—</span>
  <div id="dot"></div>
</div>

<!-- Camera view -->
<div id="view-camera" class="active">
  <div id="cv">
    <img src="/video" alt="">
    <div id="cv-lbl">Camera &middot; VIS_DEBUG annotations</div>
  </div>
  <div id="side">
    <div class="card" id="rc">
      <div class="ch">声源方向 &middot; 世界坐标 (0\xb0=正前)</div>
      <canvas id="radar" width="280" height="206"></canvas>
    </div>
    <div class="card">
      <div class="ch">系统状态</div>
      <div class="sr"><span class="sl">状态机</span><span class="sv" id="ss"></span></div>
      <div class="sr"><span class="sl">收发音</span><span class="sv" id="sp"></span></div>
      <div class="sr"><span class="sl">声源(世界)</span><span class="sv" id="sd"></span></div>
      <div class="sr"><span class="sl">方向门控</span><span class="sv" id="sg"></span></div>
      <div class="sr"><span class="sl">切换</span><span class="sv" id="sw"></span></div>
      <div class="sr"><span class="sl">头/身偏航</span><span class="sv" id="sy"></span></div>
      <div class="sr"><span class="sl">身份</span><span class="sv" id="si"></span></div>
      <div class="sr"><span class="sl">记忆</span><span class="sv" id="sm" style="font-size:10px;max-width:280px;word-break:break-all;white-space:normal"></span></div>
    </div>
  </div>
  <div id="lw">
    <div id="lh"><span>实时日志</span><span id="lc" style="margin-left:auto;color:#374151"></span></div>
    <div id="lb"></div>
    <button id="log-bottom-btn" onclick="logScrollToBottom()">&#8595;</button>
  </div>
</div>

<!-- Conversation view -->
<div id="view-conv">
  <div id="turn-list"></div>
  <div id="event-panel">
    <div id="ep-hdr">
      <span id="ep-title">全部事件</span>
      <span id="ep-count" style="color:#374151"></span>
      <span style="margin-left:auto;cursor:pointer;color:#6366f1" onclick="clearFilter()">清除过滤</span>
    </div>
    <div id="ep-list"></div>
  </div>
  <div id="timeline-wrap">
    <canvas id="tl-lanes"></canvas>
    <div id="tl-scroll"><canvas id="timeline"></canvas></div>
    <button id="tl-latest-btn" onclick="tlScrollToLatest()">▶ 滚到最新</button>
  </div>
  <div id="tl-tip"></div>
  <div id="fb-bar">
    <button id="fb-btn" onmousedown="startRec()" onmouseup="stopRec()" ontouchstart="startRec()" ontouchend="stopRec()">🎙️ 按住说反馈 <kbd style="font-size:10px;opacity:.6">[Space]</kbd></button>
    <span id="fb-status">松开后自动 ASR 归档到当前轮次</span>
  </div>
</div>

<!-- Payload modal -->
<div id="payload-modal">
  <div id="payload-box">
    <div id="pb-hdr"><h3 id="pb-title">事件详情</h3><button id="pb-close" onclick="closeModal()">✕</button></div>
    <div id="pb-body"></div>
  </div>
</div>

<script>
// ── Tab switching ──
function switchTab(name){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',['camera','conv'][i]===name));
  document.getElementById('view-camera').classList.toggle('active',name==='camera');
  document.getElementById('view-conv').classList.toggle('active',name==='conv');
  if(name==='conv') drawTimeline();
}

// ── Camera / DOA ──
const GATE=55;
const cv2=document.getElementById('radar'),cx=cv2.getContext('2d');
const W=cv2.width,H=cv2.height,CX=W/2,CY=H/2,R=Math.min(CX,CY)-16;
const y2r=d=>-(d*Math.PI/180)-Math.PI/2;
function arw(d,len,col,lw,hs=9){
  const r=y2r(d),ex=CX+Math.cos(r)*len,ey=CY+Math.sin(r)*len;
  cx.beginPath();cx.moveTo(CX,CY);cx.lineTo(ex,ey);cx.strokeStyle=col;cx.lineWidth=lw;cx.stroke();
  const a=.42;cx.beginPath();cx.moveTo(ex,ey);
  cx.lineTo(ex-hs*Math.cos(r-a),ey-hs*Math.sin(r-a));cx.lineTo(ex-hs*Math.cos(r+a),ey-hs*Math.sin(r+a));
  cx.closePath();cx.fillStyle=col;cx.fill();
}
function drawRadar(s){
  cx.clearRect(0,0,W,H);cx.fillStyle='#0c0c18';cx.beginPath();cx.arc(CX,CY,R+14,0,Math.PI*2);cx.fill();
  cx.lineWidth=1;cx.strokeStyle='#1a1a2a';
  [.35,.7,1].forEach(f=>{cx.beginPath();cx.arc(CX,CY,R*f,0,Math.PI*2);cx.stroke()});
  for(let a=0;a<360;a+=45){const r=y2r(a);cx.beginPath();cx.moveTo(CX,CY);cx.lineTo(CX+Math.cos(r)*R,CY+Math.sin(r)*R);cx.stroke()}
  cx.font='bold 11px system-ui';cx.textAlign='center';cx.textBaseline='middle';
  [[0,'前','#4ade80'],[90,'左','#94a3b8'],[-90,'右','#94a3b8'],[180,'后','#374151']].forEach(([d,t,c])=>{
    const r=y2r(d);cx.fillStyle=c;cx.fillText(t,CX+Math.cos(r)*(R+11),CY+Math.sin(r)*(R+11));
  });
  const hy=s.track_yaw||0,r1=y2r(hy-GATE),r2=y2r(hy+GATE);
  cx.beginPath();cx.moveTo(CX,CY);cx.arc(CX,CY,R*.9,r1,r2,true);cx.closePath();
  cx.fillStyle='rgba(34,211,160,.09)';cx.fill();cx.strokeStyle='rgba(34,211,160,.22)';cx.lineWidth=1;cx.stroke();
  const by=s.body_yaw_deg||0,rb=y2r(by);
  cx.setLineDash([3,4]);cx.strokeStyle='#4b5563';cx.lineWidth=1.5;
  cx.beginPath();cx.moveTo(CX,CY);cx.lineTo(CX+Math.cos(rb)*R*.75,CY+Math.sin(rb)*R*.75);cx.stroke();cx.setLineDash([]);
  arw(hy,R*.72,'#38bdf8',2.5);
  const dr=s.doa_resid_stable;
  if(dr!=null){const wd=hy+dr,fr=s.doa_fresh,col=fr?(s.doa_confident?'#22d3a0':'#f5a623'):'#374151';
    arw(wd,R*.88,col,2);const rr=y2r(wd);
    cx.beginPath();cx.arc(CX+Math.cos(rr)*R*.88,CY+Math.sin(rr)*R*.88,4,0,Math.PI*2);cx.fillStyle=col;cx.fill();
  }
  if(s.switching&&s.switch_target){const rs=y2r(s.switch_target);
    cx.beginPath();cx.arc(CX+Math.cos(rs)*R*.72,CY+Math.sin(rs)*R*.72,5,0,Math.PI*2);
    cx.strokeStyle='#f97316';cx.lineWidth=2;cx.stroke();}
  cx.beginPath();cx.arc(CX,CY,3,0,Math.PI*2);cx.fillStyle='#fff';cx.fill();
  cx.font='10px monospace';cx.textAlign='left';cx.textBaseline='alphabetic';
  const fr2=s.doa_fresh,co2=s.doa_confident,dr2=s.doa_resid_stable;
  [['‒ ‒','#4b5563','身(body)'],['——','#38bdf8','头(head)'],
   ['——',dr2!=null&&fr2?(co2?'#22d3a0':'#f5a623'):'#374151','声(world)']
  ].forEach(([sym,c,t],i)=>{cx.fillStyle=c;cx.fillText(sym+' '+t,4,H-26+i*13)});
}
const $=id=>document.getElementById(id);
const bmap={ARMED:'ba',ENGAGING:'be',TRACKING:'bt',SEEKING:'bs',SEARCHING:'bs',PLAYING:'bp',RETURNING:'br'};
function sv(id,txt,col){const e=$(id);e.textContent=txt;if(col)e.style.color=col}
function refreshCamera(s){
  const b=$('badge');b.textContent=s.state||'—';b.className='badge '+(bmap[s.state]||'ba');
  sv('ss',s.state||'—');
  sv('sp',s.speaking?'🔊 说话中':'🎙️ 收听中',s.speaking?'#f97316':'#38bdf8');
  const dr=s.doa_resid_stable,hy=s.track_yaw||0;
  if(dr!=null&&s.doa_fresh){const w=hy+dr,dir=w>5?'←左':w<-5?'右→':'↑前';
    sv('sd',(w>=0?'+':'')+w.toFixed(0)+'\xb0 '+dir+' '+(s.doa_confident?'●':'○'),s.doa_confident?'#22d3a0':'#f5a623');
  }else sv('sd',dr!=null?(hy+dr).toFixed(0)+'\xb0(旧)':'—','#374151');
  sv('sg',s.gate_open?'✓ 开放 (收音)':'✗ 静音 (门关)',s.gate_open?'#22d3a0':'#f25e6b');
  sv('sw',s.switching?s.switch_phase+' → '+s.switch_target.toFixed(0)+'\xb0':'—',s.switching?'#f97316':'#374151');
  const hv=(s.track_yaw||0).toFixed(1),bv=(s.body_yaw_deg||0).toFixed(1);
  sv('sy','头 '+(hv>=0?'+':'')+hv+'\xb0  身 '+(bv>=0?'+':'')+bv+'\xb0');
  const pn=s.identity_name||s.identity_pid||'—';
  const memS=s.identity_injected?' ✓记忆':'';
  const ownS=s.is_owner?' 👑':'';
  const clrS=s.clear_phase==='verifying'?' 🔒验证中':s.clear_phase==='confirming'?' 🔒确认中':'';
  sv('si',pn+memS+ownS+clrS,s.clear_phase?'#ef4444':s.identity_name?'#22d3a0':'#374151');
  sv('sm',s.memory_prompt||'—',s.audio_gate?'#f97316':'#6b7280');
  drawRadar(s);
}
let logSeq=0;const lb=$('lb'),MAX=400;
function lcls(t){
  if(/❌|ERROR|Traceback/.test(t))return 'le';
  if(/⚠|WARN|失败|failed/.test(t))return 'lw2';
  if(/✅|👂|🎙|🤖|就绪|启动|成功/.test(t))return 'lk';
  if(/KWS|raw=|resid=|IQR=|vad=/.test(t))return 'ld';
  return 'lm';
}
function addLog(lines){
  const f=document.createDocumentFragment();
  lines.forEach(t=>{const d=document.createElement('div');d.className='ll '+lcls(t);d.textContent=t;f.appendChild(d)});
  lb.appendChild(f);
  while(lb.children.length>MAX)lb.removeChild(lb.firstChild);
  if(logAutoScroll)lb.scrollTop=lb.scrollHeight;
  $('lc').textContent=lb.children.length+' lines';
}

let logAutoScroll=true;
const logBtn=$('log-bottom-btn');
lb.addEventListener('scroll',()=>{
  const atBot=lb.scrollTop+lb.clientHeight>=lb.scrollHeight-40;
  if(atBot){logAutoScroll=true;logBtn.style.display='none';}
  else{logAutoScroll=false;logBtn.style.display='block';}
},{passive:true});
function logScrollToBottom(){
  logAutoScroll=true;logBtn.style.display='none';
  lb.scrollTop=lb.scrollHeight;
}

// ── Conversation view ──
let convSeq=0, allEvents=[], allTurns=[], allFeedback=[], feedbackDir='';
let selectedTurnId=null, filterTurnId=null;
// timeline navigation state
let tlNodes=[], tlSelIdx=-1;

function renderTurnCard(t){
  const fbCount=allFeedback.filter(f=>f.turn_id===t.turn_id).length;
  const d=document.createElement('div');
  // Issue#4 fix: set class directly here (no separate classList.toggle pass needed)
  d.className='turn-card'+(selectedTurnId===t.turn_id?' selected':'');
  d.dataset.tid=t.turn_id;
  d.onclick=()=>selectTurn(t.turn_id);
  const dur=t.end_mono&&t.start_mono?(t.end_mono-t.start_mono).toFixed(1)+'s':'…';
  // Issue#1 fix: use turn_num (sequential 1,2,3…) instead of turn_id (event seq, has gaps)
  const numLabel=t.turn_num!=null?t.turn_num:t.turn_id;
  // context: vis/behavior events that preceded this turn
  const ctxHtml=(t.context&&t.context.length)
    ?`<div class="tc-ctx">${t.context.map(c=>`<span>${esc(c)}</span>`).join(' · ')}</div>`:
    '';
  d.innerHTML=`<div class="tc-hdr"><span class="tc-id">Turn #${numLabel}</span><span class="tc-ts">${t.start_ts||''} (${dur})</span></div>`+
    ctxHtml+
    (t.asr?`<div class="tc-row tc-asr"><span class="em">🎤</span>${esc(t.asr.slice(0,80))}</div>`:'<div class="tc-row" style="color:#374151">（等待 ASR…）</div>')+
    t.tool_calls.map(tc=>`<div class="tc-row tc-tool"><span class="em">🤖</span>${esc(tc.name)}`+(tc.output_preview?` <span style="color:#9ca3af">→ ${esc(tc.output_preview.slice(0,40))}</span>`:'')+`</div>`).join('')+
    (t.snapshot_desc?`<div class="tc-row tc-vlm"><span class="em">🖼️</span>${esc(t.snapshot_desc.slice(0,80))}</div>`:'')+
    (t.transcript?`<div class="tc-row tc-out"><span class="em">🔊</span>${esc(t.transcript.slice(0,80))}</div>`:'')+
    (fbCount?`<div class="tc-fb">📌 ${fbCount} 条反馈</div>`:'');
  return d;
}

function refreshTurnList(){
  const list=$('turn-list');
  const scrolled=list.scrollTop+list.clientHeight>=list.scrollHeight-60;
  // Issue#4 fix: full re-render ensures selected class is always in sync with selectedTurnId.
  // diff by innerHTML to avoid thrashing, but always re-create when selected state might differ.
  const existing=new Map([...list.querySelectorAll('.turn-card')].map(e=>[+e.dataset.tid,e]));
  allTurns.forEach(t=>{
    const el=existing.get(t.turn_id);
    const fresh=renderTurnCard(t);
    if(!el){
      list.appendChild(fresh);
    } else {
      // always replace if selected state changed OR content changed
      const wasSelected=el.classList.contains('selected');
      const nowSelected=(selectedTurnId===t.turn_id);
      if(wasSelected!==nowSelected||el.innerHTML!==fresh.innerHTML){
        list.replaceChild(fresh,el);
      }
    }
  });
  if(scrolled)list.scrollTop=list.scrollHeight;
}

function selectTurn(tid){
  // Clicking a card sets selectedTurnId (border highlight) but NOT filterTurnId.
  // filterTurnId is only set by explicit filter buttons if any. Scroll + card highlight only.
  selectedTurnId=tid;
  refreshTurnList();
  const turn=allTurns.find(t=>t.turn_id===tid);
  scrollCardIntoView(tid);
  // highlight this turn's events in event list without locking filter
  if(turn){
    const hlSeqs=new Set(turn.events);
    document.querySelectorAll('.ev').forEach(el=>{
      const s=+el.dataset.seq;
      el.classList.toggle('ev-hl',hlSeqs.has(s));
      el.classList.toggle('ev-dim',!hlSeqs.has(s));
    });
    const first=document.querySelector('.ev.ev-hl');
    if(first)setTimeout(()=>first.scrollIntoView({behavior:'smooth',block:'center'}),50);
    const numLabel=turn.turn_num!=null?turn.turn_num:tid;
    $('ep-title').textContent='Turn #'+numLabel+' 事件';
  }
  drawTimeline();
}
function clearFilter(){
  filterTurnId=null;selectedTurnId=null;tlSelNode=null;
  $('ep-title').textContent='全部事件';
  refreshTurnList();renderEventList();drawTimeline();
}

// Smart scroll: scroll turn-list so the card is fully visible with minimum movement
function scrollCardIntoView(tid){
  const list=$('turn-list');
  const card=document.querySelector(`.turn-card[data-tid="${tid}"]`);
  if(!card)return;
  const listRect=list.getBoundingClientRect();
  const cardRect=card.getBoundingClientRect();
  const topOff=cardRect.top-listRect.top;    // card top relative to list visible area
  const botOff=cardRect.bottom-listRect.bottom; // positive = card bottom is below list bottom
  if(topOff<0){
    // card top is hidden above — scroll up just enough
    list.scrollTop+=topOff-6;
  } else if(botOff>0){
    // card bottom is hidden below — scroll down just enough
    list.scrollTop+=botOff+6;
  }
  // if both partially visible and card is taller than list, prefer showing top
  if(cardRect.height>listRect.height) list.scrollTop+=topOff-6;
}

function getFilteredSeqs(){
  if(filterTurnId==null)return null;
  const t=allTurns.find(t=>t.turn_id===filterTurnId);
  return t?new Set(t.events):new Set();
}

// Issue#2 fix: render ALL events, highlight the ones in the selected turn with overlay
function renderEventList(){
  const hlSeqs=getFilteredSeqs(); // null=no filter, Set=highlight these
  const evs=allEvents; // always show all events
  const list=$('ep-list');
  const bot=list.scrollTop+list.clientHeight>=list.scrollHeight-40;
  list.innerHTML='';
  const f=document.createDocumentFragment();
  let firstHl=null;
  evs.forEach(e=>{
    const d=document.createElement('div');
    const isHl=hlSeqs==null||hlSeqs.has(e.seq);
    d.className=`ev role-${e.role}`+(isHl&&hlSeqs!=null?' ev-hl':'')+(hlSeqs!=null&&!isHl?' ev-dim':'');
    d.dataset.seq=e.seq;
    d.innerHTML=`<span class="es">${e.seq}</span><span class="ets">${e.ts.slice(0,12)}</span><span class="ety">${esc(e.type)}</span><span class="elb">${esc(e.label)}</span>`;
    d.onclick=()=>openModal(e);
    f.appendChild(d);
    if(isHl&&hlSeqs!=null&&firstHl==null)firstHl=d;
  });
  list.appendChild(f);
  $('ep-count').textContent=(hlSeqs!=null?hlSeqs.size+'/':'')+evs.length+' events';
  // scroll to first highlighted event
  if(firstHl)setTimeout(()=>firstHl.scrollIntoView({behavior:'smooth',block:'center'}),50);
  else if(bot)list.scrollTop=list.scrollHeight;
}

// timeline
const LANES=['user','model','tool','system'];
const LANE_LABELS={'user':'User','model':'Model','tool':'Tool','system':'Sys'};
const LANE_COLORS={'user':'#38bdf8','model':'#22d3a0','tool':'#f5a623','system':'#6366f1'};
const LANE_H=36;
const TL_LABEL_W=44;   // sticky lane-label column width
const TL_PAD_R=12, TL_PAD_T=6;
const TL_MIN_PX=24;    // minimum pixels per event (controls scroll width)
const TL_DOT_R=4;
let tlSelNode=null;
let tlAutoScroll=true;  // auto-follow latest; set false when user manually scrolls

function tlHeight(){return LANES.length*LANE_H+TL_PAD_T*2;}

function drawLaneLabels(){
  const canvas=$('tl-lanes');
  const H2=tlHeight();
  canvas.width=TL_LABEL_W; canvas.height=H2;
  const c=canvas.getContext('2d');
  c.fillStyle='#0d0d18'; c.fillRect(0,0,TL_LABEL_W,H2);
  LANES.forEach((l,i)=>{
    const y=TL_PAD_T+i*LANE_H+LANE_H/2;
    c.fillStyle=LANE_COLORS[l]; c.font='9px system-ui';
    c.textAlign='center'; c.textBaseline='middle';
    c.fillText(LANE_LABELS[l], TL_LABEL_W/2, y);
    c.strokeStyle='#1a1a2a'; c.lineWidth=1;
    c.beginPath(); c.moveTo(0,y+LANE_H/2); c.lineTo(TL_LABEL_W,y+LANE_H/2); c.stroke();
  });
}

function drawTimeline(){
  const wrap=$('tl-scroll');
  const canvas=$('timeline');
  if(!allEvents.length){canvas.width=wrap.clientWidth||400;canvas.height=tlHeight();return;}

  const displayEvs=allEvents.slice(-600);
  // compute canvas width: enough pixels per event, at least fill container
  const minW=Math.max(wrap.clientWidth||400, displayEvs.length*TL_MIN_PX+TL_PAD_R);
  const H2=tlHeight();
  canvas.width=minW; canvas.height=H2;

  const c=canvas.getContext('2d');
  c.fillStyle='#09090f'; c.fillRect(0,0,minW,H2);

  const allMono=allEvents.map(e=>e.ts_mono);
  const t0_full=Math.min(...allMono), t1_full=Math.max(...allMono);
  const span=Math.max(t1_full-t0_full,1);
  const cw=minW-TL_PAD_R;
  const tx=t=>((t-t0_full)/span)*cw;

  const filteredSeqs=filterTurnId!=null
    ?(()=>{const turn=allTurns.find(t=>t.turn_id===filterTurnId);return turn?new Set(turn.events):new Set();})()
    :null;

  // draw turn background bands
  allTurns.forEach(turn=>{
    const ts=turn.start_mono, te=turn.end_mono||t1_full;
    const x0=tx(ts), x1=tx(te);
    const isSel=(turn.turn_id===filterTurnId);
    c.fillStyle=isSel?'rgba(56,189,248,.08)':'rgba(255,255,255,.015)';
    c.fillRect(x0,0,Math.max(x1-x0,2),H2);
    if(isSel){
      c.strokeStyle='rgba(56,189,248,.3)'; c.lineWidth=1;
      c.beginPath(); c.moveTo(x0,0); c.lineTo(x0,H2); c.stroke();
    }
  });

  // draw lane horizontal lines
  LANES.forEach((_,i)=>{
    const y=TL_PAD_T+i*LANE_H+LANE_H/2;
    c.strokeStyle='#1a1a2a'; c.lineWidth=1;
    c.beginPath(); c.moveTo(0,y); c.lineTo(minW,y); c.stroke();
  });

  // draw time tick marks every ~60px
  const tickInterval=Math.max(1,(span/(cw/60)));
  c.fillStyle='#374151'; c.font='8px monospace'; c.textAlign='center'; c.textBaseline='bottom';
  for(let t=t0_full;t<=t1_full;t+=tickInterval){
    const x=tx(t);
    c.fillStyle='#1e1e2e'; c.fillRect(x-0.5,0,1,H2);
    const s=((t-t0_full));
    c.fillStyle='#4b5563'; c.fillText(s.toFixed(0)+'s',x,H2);
  }

  // draw events — no inline labels (shown on hover via tl-tip)
  tlNodes=[];
  displayEvs.forEach(e=>{
    const li=LANES.indexOf(e.role); if(li<0)return;
    const x=tx(e.ts_mono), y=TL_PAD_T+li*LANE_H+LANE_H/2;
    const inFilter=!filteredSeqs||filteredSeqs.has(e.seq);
    const isSelNode=tlSelNode&&tlSelNode.evSeq===e.seq;
    const r=isSelNode?TL_DOT_R+2:TL_DOT_R;
    c.globalAlpha=inFilter?1.0:0.18;
    c.fillStyle=inFilter?LANE_COLORS[e.role]:'#374151';
    c.beginPath(); c.arc(x,y,r,0,Math.PI*2); c.fill();
    if(isSelNode){
      c.strokeStyle='#fff'; c.lineWidth=1.5;
      c.beginPath(); c.arc(x,y,r+2,0,Math.PI*2); c.stroke();
    }
    c.globalAlpha=1.0;
    tlNodes.push({idx:tlNodes.length, evSeq:e.seq, x, y, role:e.role, laneIdx:li, event:e});
  });
  drawLaneLabels();
  // auto-scroll: if tlAutoScroll=true, always jump to rightmost
  if(tlAutoScroll) wrap.scrollLeft=wrap.scrollWidth;
}

// timeline click: move cursor only, no filter lock
$('timeline').addEventListener('click',function(ev){
  const rect=this.getBoundingClientRect();
  const mx=ev.clientX-rect.left, my=ev.clientY-rect.top;
  let best=null, bestD=Infinity;
  tlNodes.forEach(n=>{
    const d=Math.hypot(n.x-mx,n.y-my);
    if(d<bestD){bestD=d;best=n;}
  });
  if(!best||bestD>18){clearFilter();return;}
  tlSelNode=best;
  const evEl=document.querySelector(`.ev[data-seq="${best.evSeq}"]`);
  if(evEl){evEl.scrollIntoView({behavior:'smooth',block:'center'});}
  const turn=allTurns.find(t=>t.events.includes(best.evSeq));
  if(turn){scrollCardIntoView(turn.turn_id);}
  drawTimeline();
});

// hover tooltip over timeline nodes
(()=>{
  const tip=$('tl-tip');
  const canvas=$('timeline');
  canvas.addEventListener('mousemove',function(ev){
    const rect=this.getBoundingClientRect();
    const mx=ev.clientX-rect.left, my=ev.clientY-rect.top;
    let best=null, bestD=Infinity;
    tlNodes.forEach(n=>{
      const d=Math.hypot(n.x-mx,n.y-my);
      if(d<bestD){bestD=d;best=n;}
    });
    if(!best||bestD>14){tip.style.display='none';return;}
    const e=best.event;
    tip.style.display='block';
    tip.textContent=`[${e.ts.slice(0,12)}] ${e.type}\n${e.label}`;
    // read dims after display:block so offsetWidth is valid
    const tw=tip.offsetWidth||200, th=tip.offsetHeight||40;
    const tx=Math.min(ev.clientX+14, window.innerWidth-tw-8);
    const ty=Math.max(4,Math.min(ev.clientY-8, window.innerHeight-th-8));
    tip.style.left=tx+'px'; tip.style.top=ty+'px';
  });
  canvas.addEventListener('mouseleave',()=>{tip.style.display='none';});
})();

// auto-scroll control: user scrolling pauses auto-follow; button resumes
(()=>{
  const s=$('tl-scroll');
  const btn=$('tl-latest-btn');
  let userScrolling=false;
  s.addEventListener('scroll',()=>{
    const atRight=s.scrollLeft>=s.scrollWidth-s.clientWidth-30;
    if(atRight){
      tlAutoScroll=true; btn.style.display='none';
    } else {
      if(tlAutoScroll){tlAutoScroll=false;}
      btn.style.display='block';
    }
  },{passive:true});
})();

function tlScrollToLatest(){
  tlAutoScroll=true;
  $('tl-latest-btn').style.display='none';
  const s=$('tl-scroll');
  s.scrollLeft=s.scrollWidth;
}

// click blank area in event list → clear filter
$('ep-list').addEventListener('click',function(ev){
  if(ev.target===this) clearFilter();
});
// click blank area in turn-list → clear filter
$('turn-list').addEventListener('click',function(ev){
  if(ev.target===this) clearFilter();
});
// click blank area in conv view background → clear filter
$('view-conv').addEventListener('click',function(ev){
  const interactive=['turn-card','ev','fb-btn','tab'];
  if(!interactive.some(c=>ev.target.closest('.'+c)||ev.target.closest('#'+c))){
    clearFilter();
  }
});

// Issue#4: arrow key navigation on timeline
document.addEventListener('keydown',function(e){
  const convActive=document.getElementById('view-conv').classList.contains('active');
  if(e.code==='Space'&&!e.repeat&&convActive){e.preventDefault();startRec();return;}
  if(!convActive||!tlNodes.length)return;
  if(e.code==='ArrowLeft'||e.code==='ArrowRight'){
    e.preventDefault();
    const curLane=tlSelNode?tlSelNode.laneIdx:0;
    const sameLane=tlNodes.filter(n=>n.laneIdx===curLane);
    if(!sameLane.length)return;
    const curPos=tlSelNode?sameLane.findIndex(n=>n.evSeq===tlSelNode.evSeq):-1;
    const next=e.code==='ArrowRight'
      ?sameLane[Math.min(curPos+1,sameLane.length-1)]
      :sameLane[Math.max(curPos-1,0)];
    if(next){tlSelNode=next;scrollToTlNode(next);drawTimeline();}
  } else if(e.code==='ArrowUp'||e.code==='ArrowDown'){
    e.preventDefault();
    const curLane=tlSelNode?tlSelNode.laneIdx:0;
    const nextLane=e.code==='ArrowDown'?Math.min(curLane+1,LANES.length-1):Math.max(curLane-1,0);
    const laneEvs=tlNodes.filter(n=>n.laneIdx===nextLane);
    if(!laneEvs.length)return;
    const curMono=tlSelNode?tlSelNode.event.ts_mono:(allEvents[0]||{ts_mono:0}).ts_mono;
    const next=laneEvs.reduce((a,b)=>Math.abs(b.event.ts_mono-curMono)<Math.abs(a.event.ts_mono-curMono)?b:a);
    if(next){tlSelNode=next;scrollToTlNode(next);drawTimeline();}
  }
});
document.addEventListener('keyup',function(e){
  if(e.code==='Space'&&document.getElementById('view-conv').classList.contains('active')){e.preventDefault();stopRec();}
});

function scrollToTlNode(node){
  // scroll timeline canvas to show selected node
  const scroll=$('tl-scroll');
  const visLeft=scroll.scrollLeft, visRight=scroll.scrollLeft+scroll.clientWidth;
  if(node.x<visLeft+60||node.x>visRight-60){
    scroll.scrollLeft=Math.max(0,node.x-scroll.clientWidth/2);
  }
  const el=document.querySelector(`.ev[data-seq="${node.evSeq}"]`);
  if(el)el.scrollIntoView({behavior:'smooth',block:'nearest'});
  const turn=allTurns.find(t=>t.events.includes(node.evSeq));
  if(turn){
    const card=document.querySelector(`.turn-card[data-tid="${turn.turn_id}"]`);
    if(card)card.scrollIntoView({behavior:'smooth',block:'nearest'});
  }
}

// modal
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function openModal(e){
  $('pb-title').textContent=e.type;
  // 找到该事件所属的 turn，用下一个 turn 的 dt_seq 作为上界（包含本轮 tool call/output）
  let _modalDtSeq=Infinity;
  const turnIdx=allTurns.findIndex(t=>t.events&&t.events.includes(e.seq));
  if(turnIdx>=0&&turnIdx+1<allTurns.length){
    const nextT=allTurns[turnIdx+1];
    if(nextT.dt_seq!=null)_modalDtSeq=nextT.dt_seq;
  }
  // 构建内容：事件 payload + session context
  let html='<div style="margin-bottom:12px"><div style="color:#9ca3af;font-size:10px;margin-bottom:4px">Event Payload</div>';
  html+='<pre style="color:#d1d5db;margin:0">'+esc(JSON.stringify(e.payload,null,2))+'</pre></div>';
  // ── 模型视角：完整上下文重建 ──
    if(_lastSessionInstr||(_lastDisplayTranscript&&_lastDisplayTranscript.length)){
    html+='<div style="border-top:1px solid #374151;padding-top:8px">';
    html+='<div style="color:#a78bfa;font-size:10px;margin-bottom:6px;font-weight:bold">Model Context (模型视角完整上下文)</div>';
    // [System] instructions + tool definitions
    if(_lastSessionInstr){
      html+='<div style="margin-bottom:6px;padding:6px 8px;background:#1e1b2e;border-radius:4px;border-left:3px solid #a78bfa">';
      html+='<div style="color:#a78bfa;font-size:9px;margin-bottom:2px">[System]</div>';
      html+='<pre style="color:#c4b5fd;margin:0;white-space:pre-wrap;font-size:11px">'+esc(_lastSessionInstr)+'</pre>';
      html+='<div style="color:#6b7280;font-size:9px;margin-top:4px">[Tools] nod, shake_head, look_left/right/up/down, wiggle_antennas, tilt_head, end_session, take_snapshot, identify_pointed_object, remember_fact, forget_fact, clear_memory, confirm_clear</div>';
      html+='</div>';
    }
    // 对话历史
    if(_lastDisplayTranscript&&_lastDisplayTranscript.length){
      const filtered=_lastDisplayTranscript.filter(x=>x.seq<=_modalDtSeq);
      for(const e of filtered){
        if(e.role==='user'){
          const who=e.name||e.pid||'?';
          html+='<div style="margin-bottom:4px;padding:4px 8px;background:#172033;border-radius:4px;border-left:3px solid #60a5fa">';
          html+='<div style="color:#60a5fa;font-size:9px">[User] <span style="color:#6b7280">'+esc(e.ts)+' '+esc(who)+'</span></div>';
          html+='<div style="color:#93c5fd;font-size:11px">'+esc(e.text)+'</div></div>';
        }else if(e.role==='tool_call'){
          html+='<div style="margin-bottom:4px;padding:4px 8px;background:#1a1a1a;border-radius:4px;border-left:3px solid #f59e0b">';
          html+='<div style="color:#f59e0b;font-size:9px">[ToolCall] <span style="color:#6b7280">'+esc(e.ts)+'</span></div>';
          html+='<div style="color:#fbbf24;font-size:11px;font-family:monospace">'+esc(e.text)+'</div></div>';
        }else if(e.role==='tool_output'){
          html+='<div style="margin-bottom:4px;padding:4px 8px;background:#1a1a0a;border-radius:4px;border-left:3px solid #a3e635">';
          html+='<div style="color:#a3e635;font-size:9px">[ToolOutput] <span style="color:#6b7280">'+esc(e.ts)+'</span></div>';
          html+='<div style="color:#d9f99d;font-size:10px;font-family:monospace">'+esc(e.text)+'</div></div>';
        }else{
          html+='<div style="margin-bottom:4px;padding:4px 8px;background:#0f2922;border-radius:4px;border-left:3px solid #34d399">';
          html+='<div style="color:#34d399;font-size:9px">[Assistant] <span style="color:#6b7280">'+esc(e.ts)+'</span></div>';
          html+='<div style="color:#6ee7b7;font-size:11px">'+esc(e.text)+'</div></div>';
        }
      }
    }
    html+='</div>';
  }
  $('pb-body').innerHTML=html;
  $('payload-modal').classList.add('open');
}
let _lastSessionInstr=null,_lastMemPrompt=null,_lastConvLog=null,_lastDisplayTranscript=null;
function closeModal(){$('payload-modal').classList.remove('open')}
$('payload-modal').onclick=e=>{if(e.target===$('payload-modal'))closeModal()}

// ── Recording feedback ──
let mediaRec=null,recChunks=[],recTurnId=null;
async function startRec(){
  if(mediaRec)return;
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1}});
    mediaRec=new MediaRecorder(stream);recChunks=[];
    recTurnId=selectedTurnId;
    mediaRec.ondataavailable=e=>recChunks.push(e.data);
    mediaRec.start(100);
    $('fb-btn').classList.add('recording');
    $('fb-status').textContent='🔴 录音中… (松开 Space 或按钮结束)';
  }catch(e){
    $('fb-status').textContent='⚠ 麦克风不可用: '+e.message;
  }
}
async function stopRec(){
  if(!mediaRec)return;
  const rec=mediaRec; mediaRec=null;
  rec.stop();
  rec.stream.getTracks().forEach(t=>t.stop());
  rec.onstop=async()=>{
    if(!recChunks.length){$('fb-status').textContent='⚠ 未录到音频';return;}
    const blob=new Blob(recChunks,{type:'audio/webm'});
    const ab=await blob.arrayBuffer();
    const b64=btoa(String.fromCharCode(...new Uint8Array(ab)));
    $('fb-status').textContent='⏳ 识别中…';$('fb-btn').classList.remove('recording');
    try{
      const r=await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({audio_b64:b64,turn_id:recTurnId})});
      const d=await r.json();
      const loc=feedbackDir?` | 归档: ${feedbackDir}/feedback_*.jsonl`:'';
      $('fb-status').textContent='📌 已归档: '+d.transcript+loc;
    }catch(e){$('fb-status').textContent='⚠ 归档失败: '+e.message;}
  };
}

// ── Main poll loop ──
const dot=$('dot');let conn=false;
async function poll(){
  try{
    const url=`/state.json?after=${logSeq}&after_conv=${convSeq}`;
    const r=await fetch(url,{cache:'no-store'});
    if(r.ok){
      const s=await r.json();
      if(!conn){dot.classList.add('ok');conn=true}
      logSeq=s.log_seq||logSeq;
      if(s.new_logs&&s.new_logs.length)addLog(s.new_logs);
      refreshCamera(s);
      if(s.track_views)renderRegTracks(s.track_views);
      if(s.register_result)$('reg-status').textContent=s.register_result;
      // conv
      if(s.conv_events&&s.conv_events.length){
        allEvents.push(...s.conv_events);
        if(allEvents.length>2000)allEvents=allEvents.slice(-2000);
        convSeq=s.conv_seq||convSeq;
      }
      if(s.conv_turns)allTurns=s.conv_turns;
      if(s.feedback)allFeedback=s.feedback;
      if(s.feedback_dir)feedbackDir=s.feedback_dir;
      // context debug data
      if(s.session_instructions!=null)_lastSessionInstr=s.session_instructions;
      if(s.memory_prompt!=null)_lastMemPrompt=s.memory_prompt;
      if(s.conversation_log!=null)_lastConvLog=s.conversation_log;
      if(s.display_transcript!=null)_lastDisplayTranscript=s.display_transcript;
      if(document.getElementById('view-conv').classList.contains('active')){
        refreshTurnList();renderEventList();drawTimeline();
      }
    }
  }catch(e){dot.classList.remove('ok');conn=false}
  setTimeout(poll,250);
}
function renderRegTracks(tvs){
  const el=$('reg-tracks');if(!el)return;
  if(!tvs||!tvs.length){el.textContent='(无人脸)';return}
  el.innerHTML=tvs.map(v=>`<div onclick="document.getElementById('reg-tid').value=${v.track_id}" style="cursor:pointer;padding:1px 0">${v.selected?'🔵':'⚪'} T${v.track_id} <b>${v.name||'?'}</b> <span style="color:#6b7280">${v.zone||''}${v.confirmed?' ✓':''}</span></div>`).join('');
}
async function doRegister(){
  const tid=$('reg-tid').value, name=$('reg-name').value.trim();
  if(!tid||!name){$('reg-status').textContent='⚠ 填 track 号和名字';return}
  try{const r=await fetch(`/register?track_id=${tid}&name=${encodeURIComponent(name)}`);
    const d=await r.json();$('reg-status').textContent=d.msg||'已提交';$('reg-name').value='';}
  catch(e){$('reg-status').textContent='⚠ '+e.message}
}
poll();
window.addEventListener('resize',()=>{if(document.getElementById('view-conv').classList.contains('active'))drawTimeline()});
</script>
<div id="reg-panel" style="position:fixed;left:12px;bottom:12px;z-index:50;background:#1e1e2e;border:1px solid #374151;border-radius:8px;padding:8px 10px;font:12px system-ui;color:#e5e7eb;width:240px;box-shadow:0 2px 8px rgba(0,0,0,.4)">
  <div style="font-weight:600;margin-bottom:6px">🏷 注册身份(点下方人脸填入 track)</div>
  <div style="display:flex;gap:4px;margin-bottom:6px">
    <input id="reg-tid" type="number" placeholder="track" style="width:56px;background:#0f0f1a;border:1px solid #374151;color:#e5e7eb;border-radius:4px;padding:3px 5px">
    <input id="reg-name" type="text" placeholder="名字" style="flex:1;min-width:0;background:#0f0f1a;border:1px solid #374151;color:#e5e7eb;border-radius:4px;padding:3px 5px">
    <button onclick="doRegister()" style="background:#2563eb;color:#fff;border:none;border-radius:4px;padding:3px 8px;cursor:pointer">注册</button>
  </div>
  <div id="reg-tracks" style="font-size:11px;color:#9ca3af;max-height:110px;overflow-y:auto"></div>
  <div id="reg-status" style="font-size:11px;margin-top:5px;color:#9ca3af"></div>
</div>
</body></html>"""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._html()
            elif path == "/video":
                self._mjpeg()
            elif path == "/state.json":
                self._state()
            elif path == "/register":
                self._register()
            else:
                self.send_error(404)

        def _register(self):
            """UI 注册:显式给某 track 命名(绕过'谁在说话'归属)。"""
            import json as _json
            import urllib.parse as _up
            qs = dict(_up.parse_qsl(self.path.split("?", 1)[1] if "?" in self.path else ""))
            try:
                tid = int(qs.get("track_id", ""))
            except ValueError:
                tid = None
            name = (qs.get("name") or "").strip()
            if tid is None or not name:
                msg = "⚠ 需要 track_id 和 name"
            else:
                with st.lock:
                    st.register_request = {"track_id": tid, "name": name}
                    st.register_result = f"⏳ 已提交 track {tid} → {name}"
                msg = f"⏳ 已提交 track {tid} → {name}"
            body = _json.dumps({"msg": msg}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self):
            body = _VIS_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not stop.is_set():
                    data = _build_frame()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        + data + b"\r\n"
                    )
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _state(self):
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            after = 0
            after_conv = 0
            for part in qs.split("&"):
                if part.startswith("after="):
                    try:
                        after = int(part[6:])
                    except ValueError:
                        pass
                elif part.startswith("after_conv="):
                    try:
                        after_conv = int(part[11:])
                    except ValueError:
                        pass
            now = time.monotonic()
            with st.lock:
                data = {
                    "state": st.state,
                    "track_yaw": st.track_yaw,
                    "track_pitch": st.track_pitch,
                    "body_yaw_deg": st.body_yaw_deg,
                    "face_locked": st.face_locked,
                    "speaking": now < st.playback_end_estimate + 0.1,
                    "doa_resid_stable": st.doa_resid_stable,
                    "doa_confident": st.doa_confident,
                    "doa_fresh": (
                        st.doa_resid_stable is not None
                        and (now - st.doa_at) < DOA_GATE_FRESH_S
                    ),
                    "gate_open": st.dbg_gate_open,
                    "switching": st.dbg_switching,
                    "switch_phase": st.dbg_switch_phase,
                    "switch_target": st.dbg_switch_target,
                    "identity_pid": st.current_person_id,
                    "identity_name": st.current_person_name,
                    "identity_injected": st.identity_injected,
                    "is_owner": st.current_is_owner,
                    "memory_prompt": st.dbg_memory_prompt,
                    "audio_gate": st.audio_gate_closed,
                    "clear_phase": (st.clear_workflow or {}).get("phase"),
                    "user_speaking": st.user_speaking,
                    "track_views": [
                        {"track_id": v.get("track_id"), "name": v.get("name"),
                         "zone": v.get("zone"), "confirmed": v.get("confirmed"),
                         "selected": v.get("selected")}
                        for v in ((st.dbg_det or {}).get("track_views") or [])
                    ],
                    "register_result": st.register_result,
                }
            data["log_seq"] = _vis_log_seq
            data["new_logs"] = [t for s, t in _vis_log_buf if s > after]
            # 对话可视化增量字段
            data["conv_seq"] = _conv_seq
            data["conv_events"] = [e for e in _conv_events if e["seq"] > after_conv]
            data["conv_turns"] = _conv_turns[-30:]
            data["feedback"] = _feedback_notes[-50:]
            data["feedback_dir"] = SNAP_DIR
            data["instructions"] = INSTRUCTIONS  # 前端首次拿到后可缓存
            # Context debug 面板数据
            with st.lock:
                data["session_instructions"] = st.dbg_session_instructions
                data["conversation_log"] = {k: v[-20:] for k, v in st.conversation_log.items()}
                data["display_transcript"] = st.display_transcript[-100:]
            body = json.dumps(data, ensure_ascii=False, cls=_NumpyEncoder).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/feedback":
                self._feedback()
            elif path == "/debug/mock-identity":
                self._mock_identity()
            else:
                self.send_error(405)

        def _mock_identity(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                pid = body.get("pid", "mock_person_" + str(int(time.time())))
                name = body.get("name", "测试用户")
                with st.lock:
                    old_pid = st.current_person_id
                    old_name = st.current_person_name
                    st.current_person_id = pid
                    st.current_person_name = name
                    st.identity_injected = False
                    st.identity_injected_pid = None
                resp = json.dumps({"ok": True, "old_pid": old_pid, "old_name": old_name,
                                   "new_pid": pid, "new_name": name}, ensure_ascii=False)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp.encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))

        def _feedback(self):
            global _feedback_seq
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                audio_b64 = body.get("audio_b64", "")
                turn_id = body.get("turn_id")
                note_ts = time.strftime("%H:%M:%S")
                # 尝试 DashScope ASR
                transcript = ""
                try:
                    import dashscope
                    from dashscope.audio.asr import Recognition
                    import tempfile, os as _os
                    audio_bytes = base64.b64decode(audio_b64)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(audio_bytes)
                        tmp_path = f.name
                    rec = Recognition(model="paraformer-realtime-v2",
                                      format="wav", sample_rate=16000,
                                      callback=None)
                    result = rec.call(tmp_path)
                    _os.unlink(tmp_path)
                    if result and hasattr(result, "output"):
                        sentences = getattr(result.output, "sentence", []) or []
                        transcript = "".join(s.get("text", "") for s in sentences)
                except Exception as asr_e:
                    transcript = f"(ASR 不可用: {type(asr_e).__name__})"
                _feedback_seq += 1
                note = {"id": _feedback_seq, "ts": note_ts,
                        "transcript": transcript, "turn_id": turn_id,
                        "audio_b64": audio_b64}
                _feedback_notes.append(note)
                log(f"📌 反馈笔记 #{_feedback_seq}: {transcript[:60]}")
                # 持久化到磁盘
                try:
                    os.makedirs(SNAP_DIR, exist_ok=True)
                    fb_path = os.path.join(SNAP_DIR, f"feedback_{time.strftime('%Y%m%d')}.jsonl")
                    with open(fb_path, "a", encoding="utf-8") as _ff:
                        _ff.write(json.dumps(note, ensure_ascii=False) + "\n")
                except Exception as _pe:
                    log(f"⚠ 反馈写盘失败:{_pe}")
                resp = json.dumps({"ok": True, "transcript": transcript,
                                   "id": _feedback_seq}, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                self.send_error(500, str(e))

    class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    try:
        server = _Server(("0.0.0.0", port), _Handler)
        log(f"🔍 VIS_DEBUG → Dashboard: http://localhost:{port}  (浏览器打开;/video=MJPEG /state.json=状态)")
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        stop.wait()
        server.shutdown()
    except Exception as e:
        log(f"⚠ VIS_DEBUG 服务启动失败: {e}")



