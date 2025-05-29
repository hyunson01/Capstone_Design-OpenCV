import numpy as np
import os

# 카메라 매개변수 (캘리브레이션된 값 사용)
camera_id = 2
CAMERA_FILE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "camera"))
CAMERA_DIR = os.path.join(CAMERA_FILE_DIR, str(camera_id))

camera_matrix_path = os.path.join(CAMERA_DIR, "camera_matrix.npy")
dist_coeffs_path = os.path.join(CAMERA_DIR, "dist_coeffs.npy")

camera_matrix = np.load(camera_matrix_path)
dist_coeffs = np.load(dist_coeffs_path)



# 보드 크기 (cm 단위)
board_width_cm = 120
board_height_cm = 120

# 태그 정보
tag_size = 0.032  # 태그 크기 (단위: 미터)
tag_role = {      # 태그 ID와 역할 매핑
    20: "board",
}
object_points = np.array([
    [0, 0, 0],
    [tag_size, 0, 0],
    [tag_size, tag_size, 0],
    [0, tag_size, 0],
    [0, 0, tag_size],
    [tag_size, 0, tag_size],
    [tag_size, tag_size, tag_size],
    [0, tag_size, tag_size]
], dtype=np.float32)

# 격자 배열 생성
cell_size_cm = 10 # 격자 크기 (cm 단위)
grid_row = int(board_height_cm / cell_size_cm) # 세로 행 수
grid_col = int(board_width_cm / cell_size_cm) # 가로 열 수

# 격자 시각화 크기
cell_size = 50
grid_width = cell_size * grid_col
grid_height = cell_size * grid_row

# 트래커 관련련
tracker_dict={}

# 경로 시각화용 색상
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
    (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
    (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192)
]


# 명령 수행 시간
MOTION_DURATIONS = {
    "Move": 1.0,     # 전진
    "Stop": 1.0,      # 대기
    "Rotate": 1.0,     # 우회전
}
