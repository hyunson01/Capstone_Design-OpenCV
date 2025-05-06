import numpy as np
import os

# 카메라 매개변수 (캘리브레이션된 값 사용)
camera_matrix = np.array([
    [1.44881455e+03, 0, 9.80488323e+02],
    [0, 1.45151609e+03, 5.39528675e+02],
    [0, 0, 1.00000000e+00]
], dtype=np.float32)

dist_coeffs = np.array([
    [3.75921025e-03, 1.02703292e-01, -7.06313415e-05, 1.59368677e-03, -2.21477882e-01]
])

# 보드 크기 (cm 단위)
board_width_cm = 21
board_height_cm = 21

# 3D 상자 정의
tag_size = 0.04  # 태그 크기 (단위: 미터)
object_points = np.array([
    [0, 0, 0], [tag_size, 0, 0], [tag_size, tag_size, 0], [0, tag_size, 0],  # 태그 평면
    [0, 0, tag_size], [tag_size, 0, tag_size], [tag_size, tag_size, tag_size], [0, tag_size, tag_size]  # 상단 꼭짓점
], dtype=np.float32)

# 태그 상태 관리
tag_info = {}
detected_ids = set()

# 격자 배열 생성
grid_size =12
# grid_array = np.zeros((12, 12), dtype=int)
grid_width = 600
grid_height = int(grid_width * board_height_cm / board_width_cm)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 외부에 노출할 필요 없음
grid_path = os.path.abspath(os.path.join(BASE_DIR, "..", "grid.json"))


cell_size = 50

# 트래커 관련련
tracker_dict={}

# CBS 경로 및 설정
cbs_path = "D:\git\MAPF-ICBS\code"
arguments = ["--instance", "instances/map.txt", "--disjoint", "--hlsolver", "ICBS"]

# 경로 시각화용 색상
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
    (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
    (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192)
]