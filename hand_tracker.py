"""
hand_tracker.py
----------------
손 인식 전담 모듈. (mediapipe 직접 사용 - 부드럽고 자연스러운 인식, Vision Pro 스타일)

이번 버전에서 cvzone 대신 mediapipe.solutions.hands를 직접 사용하는 이유
-----------------------------------------------------------------------
cvzone의 fingersUp()은 손가락 끝(tip)과 관절(pip)의 상/하(y좌표) 또는 좌/우(x좌표,
엄지 전용) 위치를 단순 비교하는 방식이라, 손이 화면 평면상에서 돌아가 있으면
(예: 살짝 기울어진 주먹, 회전된 손 모양) 오탐이 잦다.

여기서는 "손목(wrist)에서 각 손가락 끝까지의 거리"와 "손목에서 각 관절까지의 거리"를
비교하는 방사형(radial) 거리 기반 판별을 직접 구현했다. 손가락이 펴지면 손목에서
멀어지고, 접히면 손목 쪽으로 가까워지는 원리를 이용하기 때문에, 손 전체가 이미지
평면에서 어떤 각도로 회전해 있어도 비교적 안정적으로 동작한다.

핵심 개선점
-----------
1. mediapipe Hands를 직접 사용해 21개 손 랜드마크를 얻는다.
2. 손목 기준 방사형 거리 비교로 손가락 펴짐 여부를 판별한다 (회전에 강함, 주먹=팬 판정용).
3. One Euro Filter로 손 좌표 / 핀치 거리를 필터링해서 프레임 간 떨림(jitter)을 제거한다.
4. 방향은 이분법이 아니라 연속값(offset_x/y)으로 반환해 부드러운 가속을 만든다.
5. 회전<->팬 전환도 0/1이 아니라 연속값(control_strength)으로 제공해 서서히 전환된다.
6. 줌은 엄지(4)-검지(8) 거리를 이용한 "적응형 중간값" 방식으로 판별한다 (아래 설명).

동작 모드 (3가지 손 제스처로 구분)
----------------------------------
- 손 전체를 편 상태 [1,1,1,1,1]     : 회전
- 주먹 [0,0,0,0,0]                  : 팬(카메라 상하좌우 이동)
- 엄지+검지만 편 상태 [1,1,0,0,0]   : 줌 (이 포즈일 때만 활성화, 회전/팬은 자동으로 꺼짐)

세 모드는 손가락 패턴을 즉시 이분법으로 스위칭하지 않고, 각각 0~1 게이트 값을
CONTROL_EASE로 서서히 수렴시켜서 제스처가 바뀌는 순간에도 급정지/급출발 없이
부드럽게 넘어가도록 했다.

줌 로직 (엄지+검지 포즈 + 핀치 거리의 적응형 중간값 기준)
-----------------------------------------------------------
줌은 "엄지+검지만 펴져 있는 포즈"([1,1,0,0,0])일 때만 활성화된다 (zoom_gate).
이 포즈가 아니면(손 전체를 편 상태로 회전 중이거나 주먹 상태) 엄지-검지 사이 거리가
아무리 변해도 줌에 전혀 반영되지 않는다. 그 덕분에 회전 중 손가락이 자연스럽게
벌어지면서 발생하던 오탐 줌이 근본적으로 사라진다.

활성화된 상태에서는 엄지(4)-검지(8) 사이의 거리를 측정한다. 이 거리에는 사람마다,
그리고 카메라와의 거리에 따라 "최대로 벌릴 수 있는 거리"와 "최소로 좁힐 수 있는
거리"가 다르므로, 고정된 픽셀 임계값을 쓰지 않고 최근 관측된 최댓값/최솟값을
실시간으로 추적해서 그 중간값(pinch_mid)을 기준으로 삼는다. 캘리브레이션(최댓값/
최솟값 갱신)도 실제로 이 포즈일 때만 이루어지도록 해서, 다른 제스처 중의 손가락
거리에 영향받지 않는다.

    - pinch_max: 관측된 거리보다 크면 즉시 갱신, 아니면 서서히 감소(decay)
    - pinch_min: 관측된 거리보다 작으면 즉시 갱신, 아니면 서서히 증가(decay)
    - pinch_mid = (pinch_max + pinch_min) / 2

    - 현재 거리 > pinch_mid  ->  줌 인  (손가락을 벌리는 중)
    - 현재 거리 < pinch_mid  ->  줌 아웃 (손가락을 좁히는 중)

단순 부호 판별에 그치지 않고, 중간값에서 얼마나 떨어져 있는지를 -1.0~1.0 사이의
연속값으로 정규화한 뒤 zoom_gate를 곱해서 최종 zoom_strength로 반환한다. 그래야
"살짝 벌림 = 천천히 줌인", "완전히 벌림 = 빠르게 줌인"처럼 속도감 있는 제어가
가능하고, 동시에 포즈가 풀리는 순간 줌도 gate를 따라 부드럽게 0으로 잦아든다.
중간값 근처에는 작은 Dead Zone(PINCH_DEAD_ZONE)을 둬서 떨림에 의한 깜빡임도 막는다.

main_controller.py 에서 이 모듈을 import 해서 사용한다.
"""

import math
import time

import cv2
import mediapipe as mp


class OneEuroFilter:
    """
    실시간 신호(손 좌표, 핀치 거리 등)를 낮은 지연으로 부드럽게 만드는 필터.
    느린 움직임은 강하게 스무딩하고, 빠른 움직임은 덜 스무딩해서
    "부드러우면서도 민첩한" 반응을 만든다.
    """

    def __init__(self, min_cutoff=1.2, beta=0.4, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x, t=None):
        t = time.time() if t is None else t

        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def smoothstep(edge0, edge1, x):
    """0~1 사이를 부드러운 S자 곡선으로 보간 (선형보다 자연스러운 가속/감속)."""
    if edge0 == edge1:
        return 0.0 if x < edge0 else 1.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3 - 2 * t)


class HandTracker:
    DEAD_ZONE = 60          # 방향 반응이 시작되는 중심 Dead Zone (px)
    FULL_RANGE = 220        # 이 거리(px)에서 최고 속도(반응 강도 1.0)에 도달

    SIZE_RATIO = 0.55       # 손 크기가 최근 최댓값 대비 이 비율보다 작아지면 "주먹"으로 판정
    SIZE_DECAY = 0.995

    CONTROL_EASE = 0.18     # 주먹<->정상(=팬<->회전) 전환을 부드럽게 만드는 완화 계수

    # 방사형 거리 판별 시, tip이 관절보다 이 배율 이상 멀어야 "펴짐"으로 인정 (검지~소지)
    FINGER_EXTEND_RATIO = 1.15
    # 엄지는 손목 기준 거리로는 거의 항상 "펴짐"으로 오탐되므로 별도 기준 사용:
    # 엄지 끝(4)이 새끼손가락 MCP(17)로부터, 엄지 MCP(2)-새끼손가락 MCP(17) 거리보다
    # 이 배율 이상 멀어야 "펴짐"으로 인정한다.
    THUMB_EXTEND_RATIO = 1.15

    # --- 핀치(엄지-검지) 줌 관련 ---
    PINCH_RANGE_DECAY = 0.995   # max는 이 비율로 서서히 감소, min은 이 비율의 역수로 서서히 증가
    PINCH_MIN_RANGE = 25        # max-min이 이 값보다 작아지지 않도록(초기/캘리브레이션 전 보호용, px)
    # 중간값(pinch_mid) 기준 이 비율(0~1) 이내는 줌 없음(떨림 방지용 중립 구간).
    # 이제 줌 자체가 "엄지+검지 포즈"일 때만 활성화(zoom_gate)되므로, 회전 중
    # 오탐 문제는 게이팅이 해결한다. 이 값은 그 포즈 안에서의 손 떨림만 걸러주면
    # 되므로 다시 작게 잡았다. 필요하면 조절하세요.
    PINCH_DEAD_ZONE = 0.12

    # 제스처 게이트(회전/팬/줌)가 전환될 때 부드럽게 수렴하는 속도.
    # 값을 작게 하면 제스처 전환이 더 느긋하고 부드럽게, 크게 하면 더 즉각적으로 바뀐다.
    GATE_EASE = 0.18

    # mediapipe 랜드마크 인덱스: [엄지, 검지, 중지, 약지, 소지]
    _TIP_IDS = [4, 8, 12, 16, 20]
    _JOINT_IDS = [2, 6, 10, 14, 18]  # 엄지는 MCP(2), 나머지는 PIP 관절

    def __init__(self, camera_index=0, detection_con=0.7, tracking_con=0.7, max_hands=1):
        self.cap = cv2.VideoCapture(camera_index)

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self.hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=tracking_con,
        )

        self._filter_x = OneEuroFilter(min_cutoff=1.2, beta=0.5)
        self._filter_y = OneEuroFilter(min_cutoff=1.2, beta=0.5)
        self._filter_pinch = OneEuroFilter(min_cutoff=1.5, beta=0.6)

        self._max_hand_size = 0
        self._control_strength = 1.0  # 1.0 = 손을 편 상태(회전), 0.0 = 주먹(팬) 쪽 성향
        self._zoom_gate = 0.0         # 1.0 = 엄지+검지 포즈(줌 활성), 0.0 = 그 외(줌 비활성)

        # 핀치 거리의 관측된 최댓값/최솟값 (적응형 캘리브레이션, 엄지+검지 포즈일 때만 갱신)
        self._pinch_max = None
        self._pinch_min = None

    def read(self):
        success, img = self.cap.read()
        if not success:
            return False, None
        img = cv2.flip(img, 1)
        return True, img

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _fingers_up(self, lm):
        """
        검지~소지: 손목(0번) 기준 방사형 거리 비교로 판별.
        엄지: 새끼손가락 MCP(17) 기준 거리 비교로 판별.
        손 전체가 이미지 평면에서 회전해 있어도 거리 기반 기준은 유지되므로
        위/아래 좌표 비교 방식보다 훨씬 안정적으로 동작한다. (주먹=팬 판정에 사용)

        Returns
        -------
        list[int] : [엄지, 검지, 중지, 약지, 소지] 순서, 각각 1(펴짐)/0(접힘)
        """
        wrist = lm[0]
        pinky_mcp = lm[17]

        thumb_tip_d = self._dist(lm[4], pinky_mcp)
        thumb_mcp_d = self._dist(lm[2], pinky_mcp)
        thumb_up = 1 if thumb_tip_d > thumb_mcp_d * self.THUMB_EXTEND_RATIO else 0

        fingers = [thumb_up]
        for tip_i, joint_i in zip(self._TIP_IDS[1:], self._JOINT_IDS[1:]):
            tip_d = self._dist(lm[tip_i], wrist)
            joint_d = self._dist(lm[joint_i], wrist)
            fingers.append(1 if tip_d > joint_d * self.FINGER_EXTEND_RATIO else 0)
        return fingers

    def _update_pinch_zoom(self, lm, now, calibrate):
        """
        엄지(4)-검지(8) 거리를 측정하고, 그 중간값(pinch_mid)을 기준으로
        -1.0(줌아웃)~1.0(줌인) 사이의 연속값을 반환한다.

        calibrate=True (엄지+검지 포즈일 때)일 때만 관측된 최댓값/최솟값을 갱신한다.
        다른 제스처(회전/팬) 중에는 손가락 거리가 계속 변해도 캘리브레이션에
        영향을 주지 않도록 하기 위함이다.
        """
        raw_dist = self._dist(lm[4], lm[8])
        pinch_dist = self._filter_pinch.filter(raw_dist, now)

        if self._pinch_max is None:
            # 최초 프레임: 관측값을 기준점으로 초기화 (포즈 여부와 무관하게 한 번은 필요)
            self._pinch_max = pinch_dist
            self._pinch_min = pinch_dist
        elif calibrate:
            # 최댓값: 더 큰 값이 나오면 즉시 갱신, 아니면 서서히 감소 (오래된 최댓값이 자연히 잊혀짐)
            self._pinch_max = max(pinch_dist, self._pinch_max * self.PINCH_RANGE_DECAY)
            # 최솟값: 더 작은 값이 나오면 즉시 갱신, 아니면 서서히 증가
            self._pinch_min = min(pinch_dist, self._pinch_min / self.PINCH_RANGE_DECAY)

        pinch_range = max(self._pinch_max - self._pinch_min, self.PINCH_MIN_RANGE)
        pinch_mid = (self._pinch_max + self._pinch_min) / 2.0

        # 중간값 기준 -1.0~1.0 정규화 (중간값보다 크면 +, 작으면 -)
        raw_zoom = (pinch_dist - pinch_mid) / (pinch_range / 2.0)
        raw_zoom = max(-1.0, min(1.0, raw_zoom))

        # 중간값 근처 Dead Zone: 손 떨림으로 인한 줌인/줌아웃 깜빡임 방지
        if abs(raw_zoom) < self.PINCH_DEAD_ZONE:
            zoom_strength = 0.0
        else:
            sign = 1.0 if raw_zoom > 0 else -1.0
            mag = (abs(raw_zoom) - self.PINCH_DEAD_ZONE) / (1.0 - self.PINCH_DEAD_ZONE)
            zoom_strength = sign * mag

        return zoom_strength, pinch_dist, pinch_mid

    def process(self, img, draw=True):
        """
        Returns
        -------
        img : 손 랜드마크가 그려진 이미지
        info : dict
            {
              'found': bool,
              'control_strength': float,  # 0(주먹->팬)~1(손을 폄->회전)
              'offset_x': float,          # -1.0~1.0, 화면 중심 기준 정규화된 x 편차
              'offset_y': float,          # -1.0~1.0, y 편차
              'direction': str,           # 표시용 라벨 (모드 + 방향)
              'zoom_strength': float,     # -1.0(줌아웃)~1.0(줌인), 0은 중립
              'pinch_dist': float,        # 현재 엄지-검지 거리 (px, 필터링됨)
              'pinch_mid': float,         # 현재 줌 기준 중간값 (px)
              'center': (cx, cy) or None,
            }
        """
        h, w, _ = img.shape
        cx_screen, cy_screen = w // 2, h // 2
        now = time.time()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)

        cv2.rectangle(
            img,
            (cx_screen - self.DEAD_ZONE, cy_screen - self.DEAD_ZONE),
            (cx_screen + self.DEAD_ZONE, cy_screen + self.DEAD_ZONE),
            (255, 255, 255), 1
        )

        info = {
            'found': False,
            'control_strength': 0.0,
            'rotate_strength': 0.0,
            'pan_strength': 0.0,
            'zoom_gate': 0.0,
            'offset_x': 0.0,
            'offset_y': 0.0,
            'direction': 'CENTER',
            'zoom_strength': 0.0,
            'pinch_dist': 0.0,
            'pinch_mid': 0.0,
            'center': None,
            'fingers': None,
        }

        if not result.multi_hand_landmarks:
            # 손이 안 보이면 세 모드 모두 서서히 0으로 (급정지 없이 부드럽게 멈춤)
            self._control_strength += (0.0 - self._control_strength) * self.CONTROL_EASE
            self._zoom_gate += (0.0 - self._zoom_gate) * self.GATE_EASE
            info['control_strength'] = self._control_strength
            info['zoom_gate'] = self._zoom_gate
            return img, info

        hand_landmarks = result.multi_hand_landmarks[0]
        if draw:
            self._mp_draw.draw_landmarks(
                img, hand_landmarks, self._mp_hands.HAND_CONNECTIONS
            )

        lm = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]

        xs = [p[0] for p in lm]
        ys = [p[1] for p in lm]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        hand_size = max(1, (x_max - x_min) * (y_max - y_min))
        cx_raw = (x_min + x_max) // 2
        cy_raw = (y_min + y_max) // 2

        # --- 좌표 스무딩 (One Euro Filter) ---
        cx = self._filter_x.filter(cx_raw, now)
        cy = self._filter_y.filter(cy_raw, now)

        # --- 손 크기 기반 보조 주먹 판별 (각도 회전에 덜 민감) ---
        if hand_size > self._max_hand_size:
            self._max_hand_size = hand_size
        else:
            self._max_hand_size *= self.SIZE_DECAY
        is_small = (self._max_hand_size > 0) and (hand_size < self._max_hand_size * self.SIZE_RATIO)

        # --- 손가락 패턴: 세 가지 제스처 판별 (주먹=팬, 엄지+검지만=줌) ---
        fingers = self._fingers_up(lm)
        is_all_closed = fingers == [0, 0, 0, 0, 0]
        is_fist_frame = is_all_closed or is_small
        is_pinch_pose = (fingers == [1, 1, 0, 0, 0])  # 엄지+검지만 편 상태 -> 줌 포즈

        # --- 회전<->팬 성향: 즉시 0/1이 아니라 서서히 완화(ease) ---
        # (엄지+검지 포즈는 주먹이 아니므로 여기선 "편 손" 쪽으로 향하지만,
        #  아래 movement_gate가 곱해져서 실제 회전/팬에는 반영되지 않는다.)
        target_strength = 0.0 if is_fist_frame else 1.0
        self._control_strength += (target_strength - self._control_strength) * self.CONTROL_EASE

        # --- 줌 게이트: 엄지+검지 포즈일 때만 서서히 1로, 아니면 서서히 0으로 ---
        target_zoom_gate = 1.0 if is_pinch_pose else 0.0
        self._zoom_gate += (target_zoom_gate - self._zoom_gate) * self.GATE_EASE
        movement_gate = 1.0 - self._zoom_gate  # 줌 포즈일수록 회전/팬은 꺼진다

        # 회전/팬 최종 강도 (줌 포즈 중에는 movement_gate가 0에 가까워지며 자동으로 꺼짐)
        rotate_strength = self._control_strength * movement_gate
        pan_strength = (1.0 - self._control_strength) * movement_gate

        # --- 줌: 엄지-검지 거리의 적응형 중간값 기준 (엄지+검지 포즈일 때만 캘리브레이션 갱신) ---
        raw_zoom_strength, pinch_dist, pinch_mid = self._update_pinch_zoom(lm, now, calibrate=is_pinch_pose)
        zoom_strength = raw_zoom_strength * self._zoom_gate  # 포즈가 아니면 자동으로 0에 수렴

        # --- 연속값 기반 방향 오프셋 (부드러운 가속용, 회전/팬 공용) ---
        dx = cx - cx_screen
        dy = cy - cy_screen
        mag_x = smoothstep(self.DEAD_ZONE, self.FULL_RANGE, abs(dx))
        mag_y = smoothstep(self.DEAD_ZONE, self.FULL_RANGE, abs(dy))
        offset_x = math.copysign(mag_x, dx)
        offset_y = math.copysign(mag_y, dy)

        # --- 표시용 라벨: 모드(ROTATE/PAN/ZOOM) + 방향 ---
        if self._zoom_gate >= 0.5:
            mode_label = 'ZOOM'
        elif self._control_strength >= 0.5:
            mode_label = 'ROTATE'
        else:
            mode_label = 'PAN'
        direction = 'CENTER'
        if offset_x < -0.05:
            direction = 'LEFT'
        elif offset_x > 0.05:
            direction = 'RIGHT'
        if offset_y < -0.05:
            direction = 'UP' if direction == 'CENTER' else direction + '_UP'
        elif offset_y > 0.05:
            direction = 'DOWN' if direction == 'CENTER' else direction + '_DOWN'
        direction = f'{mode_label} {direction}'

        dot_color = (
            int(255 * (1 - self._control_strength)),
            int(120 + 100 * self._control_strength),
            int(255 * self._control_strength)
        )
        cv2.circle(img, (int(cx), int(cy)), 6 + int(4 * (1 - self._control_strength)), dot_color, -1)

        # 엄지-검지 라인 표시 (줌 방향에 따라 색이 바뀜: 초록=줌인, 빨강=줌아웃)
        thumb_pt, index_pt = lm[4], lm[8]
        if zoom_strength > 0:
            pinch_color = (0, 255, 0)
        elif zoom_strength < 0:
            pinch_color = (0, 0, 255)
        else:
            pinch_color = (200, 200, 200)
        cv2.line(img, thumb_pt, index_pt, pinch_color, 2)
        cv2.circle(img, thumb_pt, 5, pinch_color, -1)
        cv2.circle(img, index_pt, 5, pinch_color, -1)

        info.update({
            'found': True,
            'control_strength': self._control_strength,
            'rotate_strength': rotate_strength,
            'pan_strength': pan_strength,
            'zoom_gate': self._zoom_gate,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'direction': direction,
            'zoom_strength': zoom_strength,
            'pinch_dist': pinch_dist,
            'pinch_mid': pinch_mid,
            'center': (cx, cy),
            'fingers': fingers,
        })
        return img, info

    def draw_ui(self, img, info):
        rotate_pct = int(info['rotate_strength'] * 100)
        pan_pct = int(info['pan_strength'] * 100)
        zoom_gate_pct = int(info['zoom_gate'] * 100)
        color = (0, 255, 0) if info['rotate_strength'] > 0.5 else (0, 165, 255)
        cv2.putText(
            img,
            f'{info["direction"]}  (rotate {rotate_pct}% / pan {pan_pct}% / zoom {zoom_gate_pct}%)',
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )

        zoom_label = 'ZOOM IN' if info['zoom_strength'] > 0 else ('ZOOM OUT' if info['zoom_strength'] < 0 else 'ZOOM -')
        zoom_color = (0, 255, 0) if info['zoom_strength'] > 0 else ((0, 0, 255) if info['zoom_strength'] < 0 else (200, 200, 200))
        cv2.putText(img,
                    f'{zoom_label} ({info["zoom_strength"]:+.2f})  dist={info["pinch_dist"]:.0f} mid={info["pinch_mid"]:.0f}',
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, zoom_color, 2)

        if info.get('fingers') is not None:
            f = info['fingers']
            labels = ['T', 'I', 'M', 'R', 'P']
            fingers_text = ' '.join(f'{lab}:{v}' for lab, v in zip(labels, f))
            cv2.putText(img, fingers_text, (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # 줌 강도 바 (중앙이 0, 왼쪽=줌아웃, 오른쪽=줌인)
        bar_x, bar_y, bar_w, bar_h = 20, 125, 200, 12
        bar_mid_x = bar_x + bar_w // 2
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        cv2.line(img, (bar_mid_x, bar_y), (bar_mid_x, bar_y + bar_h), (255, 255, 255), 1)
        fill_w = int((bar_w / 2) * info['zoom_strength'])
        if fill_w >= 0:
            cv2.rectangle(img, (bar_mid_x, bar_y), (bar_mid_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
        else:
            cv2.rectangle(img, (bar_mid_x + fill_w, bar_y), (bar_mid_x, bar_y + bar_h), (0, 0, 255), -1)

        # 회전 강도 바 (줌 포즈 중에는 자동으로 줄어듦)
        bar2_y = 150
        cv2.rectangle(img, (bar_x, bar2_y), (bar_x + bar_w, bar2_y + bar_h), (255, 255, 255), 1)
        fill2_w = int(bar_w * info['rotate_strength'])
        cv2.rectangle(img, (bar_x, bar2_y), (bar_x + fill2_w, bar2_y + bar_h), color, -1)
        return img

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    tracker = HandTracker()
    while True:
        ok, img = tracker.read()
        if not ok:
            break
        img, info = tracker.process(img)
        img = tracker.draw_ui(img, info)
        cv2.imshow("Hand Tracker Debug", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    tracker.release()