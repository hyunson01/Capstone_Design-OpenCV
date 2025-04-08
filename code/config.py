import numpy as np

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

grid_path = r"D:\git\Capstone_temp\grid.json"

cell_size = 50

# 트래커 관련련
tracker_dict={}

# CBS 경로 및 설정
cbs_path = "D:\git\MAPF-ICBS\code"
arguments = ["--instance", "instances/map.txt", "--disjoint", "--hlsolver", "ICBS"]