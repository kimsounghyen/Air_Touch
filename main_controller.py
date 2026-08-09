"""
main_controller.py
-------------------
hand_tracker.py가 손 제스처를 세 모드(회전/팬/줌)로 게이팅한 연속값
(rotate_strength, pan_strength, zoom_strength)을 받아서, 실제 회전/팬/줌에
"부드러운 관성과 완화(easing)"를 입혀 적용하는 실행 코드.

동작 모드 (hand_tracker.py에서 손가락 패턴으로 판별)
-----------------------------------------------------
- 손 전체를 편 상태     : 물체를 회전 (ctr.rotate)      -> rotate_strength가 1에 가까워짐
- 주먹                  : 카메라 상하좌우 이동/팬 (ctr.translate) -> pan_strength가 1에 가까워짐
- 엄지+검지만 편 상태   : 줌 인/아웃 (ctr.scale)         -> 이때 rotate/pan_strength는 자동으로 0

세 모드는 hand_tracker.py에서 이미 서로 배타적으로(그러나 부드럽게) 게이팅되어
넘어오므로, 여기서는 각 값을 그대로 속도 목표치에 곱해서 이징만 적용하면 된다.

줌 로직 요약 (hand_tracker.py 참고)
------------------------------------
줌은 "엄지+검지만 편 상태"일 때만 활성화된다. 활성화된 동안 엄지-검지 거리의
"관측된 최댓값/최솟값"을 실시간으로 추적해서 그 중간값(mid)을 기준으로:
    - 현재 거리 > mid  ->  줌 인  (zoom_strength > 0)
    - 현재 거리 < mid  ->  줌 아웃 (zoom_strength < 0)
중간값에서 멀어질수록 zoom_strength의 절댓값이 커져서, 살짝 벌리면 천천히,
크게 벌리면 빠르게 줌인되는 식으로 속도감 있게 동작한다.

부드러움을 만드는 장치
----------------------
1. 회전/팬/줌 속도를 매 프레임 즉시 적용하지 않고, 목표 속도로 서서히 수렴시킨다.
2. 손을 펴고 접는 정도(control_strength)에 따라 회전 강도와 팬 강도가
   서로 반대로(rotate_strength = s, pan_strength = 1 - s) 서서히 전환된다.
3. 줌도 이벤트가 아니라 매 프레임 연속값으로 들어오므로, 속도 기반 이징으로
   부드럽게 확대/축소된다.

실행:
    python main_controller.py
종료:
    OpenCV 창에서 'q' 또는 Open3D 창에서 'Q'
"""

import cv2
import open3d as o3d

from hand_tracker import HandTracker

MODEL_PATH = "Cable_Holder.obj"

MAX_ROTATE_SPEED = 10.0     # 오프셋 1.0(최고 강도)일 때의 최대 회전 속도
VELOCITY_EASE = 0.3         # 회전 속도가 목표치로 수렴하는 속도 (작을수록 더 부드럽고 느리게 반응)

MAX_PAN_SPEED = 8.0         # 오프셋 1.0(최고 강도)일 때의 최대 팬(카메라 이동) 속도
PAN_VELOCITY_EASE = 0.3     # 팬 속도가 목표치로 수렴하는 속도
# Open3D 버전/환경에 따라 팬 방향이 반대로 느껴지면 아래 두 값의 부호를 바꾸세요.
PAN_SIGN_X = 1
PAN_SIGN_Y = 1

MAX_ZOOM_SPEED = 0.5        # zoom_strength가 1.0(최대 줌인)일 때의 최대 scale 변화 속도
ZOOM_VELOCITY_EASE = 0.25   # 줌 속도가 목표치로 수렴하는 속도 (작을수록 더 부드러운 '스프링' 느낌)
# Open3D 버전에 따라 scale() 부호가 반대로 느껴지면(줄이면 커짐 등) 이 값을 다시 1로 바꾸세요.
ZOOM_SIGN = -1


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    if len(mesh.vertices) == 0:
        raise FileNotFoundError(f"'{path}' 모델을 불러오지 못했습니다. 경로를 확인하세요.")
    return mesh


def main():
    mesh = load_mesh(MODEL_PATH)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("3D Control Center")
    vis.add_geometry(mesh)

    tracker = HandTracker(camera_index=0, detection_con=0.7, tracking_con=0.7, max_hands=1)

    state = {
        "running": True,
        "vel_x": 0.0, "vel_y": 0.0,        # 현재 회전 속도
        "vel_px": 0.0, "vel_py": 0.0,      # 현재 팬 속도
        "vel_zoom": 0.0,                   # 현재 줌 속도
    }

    def close_all(vis_):
        state["running"] = False
        tracker.release()
        return False

    vis.register_key_callback(ord("Q"), close_all)

    def update_frame(vis_):
        if not state["running"]:
            return False

        ok, img = tracker.read()
        if not ok:
            return False

        img, info = tracker.process(img, draw=True)
        img = tracker.draw_ui(img, info)

        ctr = vis_.get_view_control()

        # rotate_strength / pan_strength는 hand_tracker.py에서 이미 게이팅된 값이다.
        # (엄지+검지 줌 포즈일 때는 둘 다 자동으로 0에 가까워진다)
        rotate_strength = info['rotate_strength']
        pan_strength = info['pan_strength']

        # --- 1. 회전: 손을 편 정도만큼 목표 속도로 부드럽게 수렴 ---
        target_vx = info['offset_x'] * MAX_ROTATE_SPEED * rotate_strength
        target_vy = info['offset_y'] * MAX_ROTATE_SPEED * rotate_strength

        state["vel_x"] += (target_vx - state["vel_x"]) * VELOCITY_EASE
        state["vel_y"] += (target_vy - state["vel_y"]) * VELOCITY_EASE

        if abs(state["vel_x"]) > 0.02 or abs(state["vel_y"]) > 0.02:
            ctr.rotate(state["vel_x"], state["vel_y"])

        # --- 2. 팬: 주먹을 쥔 정도만큼 목표 속도로 부드럽게 수렴 (카메라 구도 상하좌우 이동) ---
        target_px = info['offset_x'] * MAX_PAN_SPEED * pan_strength * PAN_SIGN_X
        target_py = info['offset_y'] * MAX_PAN_SPEED * pan_strength * PAN_SIGN_Y

        state["vel_px"] += (target_px - state["vel_px"]) * PAN_VELOCITY_EASE
        state["vel_py"] += (target_py - state["vel_py"]) * PAN_VELOCITY_EASE

        if abs(state["vel_px"]) > 0.02 or abs(state["vel_py"]) > 0.02:
            ctr.translate(state["vel_px"], state["vel_py"])

        # --- 3. 줌: 엄지-검지 거리의 중간값 기준 연속값을 목표 속도로 부드럽게 수렴 ---
        target_zoom_vel = info['zoom_strength'] * MAX_ZOOM_SPEED * ZOOM_SIGN
        state["vel_zoom"] += (target_zoom_vel - state["vel_zoom"]) * ZOOM_VELOCITY_EASE

        if abs(state["vel_zoom"]) > 0.001:
            ctr.scale(state["vel_zoom"])

        cv2.imshow("Hand Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return close_all(vis_)

        return True

    vis.register_animation_callback(update_frame)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()