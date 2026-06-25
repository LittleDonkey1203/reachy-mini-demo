# -*- coding: utf-8 -*-
"""声源-视觉感知融合 — 多模态匹配。"""

from voice.config import FOV_X_DEG


def select_face_by_doa(all_faces: list[dict], doa_resid: float,
                       track_yaw: float, body_yaw: float,
                       fov_deg: float = FOV_X_DEG) -> int | None:
    """从多张人脸中选出最接近 DOA 声源方向的脸，返回 all_faces 中的索引。

    DOA resid 是相对于身体正前方的声源角度(正=左)。
    摄像头在头上，头朝向 track_yaw(正=左)，身体朝向 body_yaw(正=左)。
    声源在摄像头画面中的预期 u 坐标:
      doa_in_camera = (body_yaw + resid) - track_yaw
      expected_u = 0.5 - doa_in_camera / fov_deg
    """
    if not all_faces or len(all_faces) <= 1:
        return None
    doa_in_camera = (body_yaw + doa_resid) - track_yaw
    expected_u = 0.5 - doa_in_camera / fov_deg
    expected_u = max(0.0, min(1.0, expected_u))
    best_idx = 0
    best_dist = abs(all_faces[0]["u"] - expected_u)
    for i in range(1, len(all_faces)):
        d = abs(all_faces[i]["u"] - expected_u)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx
