# new_board.py

import numpy as np
import cv2

def get_cm_per_pixel(tag_corners, tag_length_cm):
    """
    AprilTag 한 변의 길이(cm)와 네 점의 픽셀 좌표로부터 cm/px 비율 계산
    """
    top_left, top_right, bottom_right, bottom_left = tag_corners
    width_px = np.linalg.norm(top_right - top_left)
    height_px = np.linalg.norm(bottom_left - top_left)
    avg_px = (width_px + height_px) / 2
    return tag_length_cm / avg_px

def get_board_corners_from_tag(tag_corners, tag_length_cm=5.0, board_width_cm=120.0, board_height_cm=90.0):
    """
    하나의 AprilTag 좌표와 크기를 기준으로 전체 경기장 사각형 좌표 생성
    """
    # 태그의 꼭짓점
    top_left, top_right, bottom_right, bottom_left = tag_corners

    # 태그의 실제 중심 좌표 계산 (픽셀 단위)
    tag_center = np.mean(tag_corners, axis=0)

    # 픽셀 단위 변환 비율 계산
    cm_per_px = get_cm_per_pixel(tag_corners, tag_length_cm)
    px_per_cm = 1 / cm_per_px

    # 좌표계 기준: AprilTag는 왼쪽 아래에 있으므로 이 기준점에서 오른쪽 위 방향으로 생성
    origin = tag_center

    # 경기장의 네 꼭짓점 (픽셀 좌표 기준)
    top_left_corner     = origin + np.array([0, -board_height_cm * px_per_cm])
    top_right_corner    = origin + np.array([board_width_cm * px_per_cm, -board_height_cm * px_per_cm])
    bottom_right_corner = origin + np.array([board_width_cm * px_per_cm, 0])
    bottom_left_corner  = origin

    board_corners = np.array([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner], dtype=np.float32)

    return board_corners, cm_per_px

def draw_board_outline(frame, board_corners):
    """
    보정된 경기장 외곽선 시각화
    """
    board_corners_int = board_corners.astype(int)
    cv2.polylines(frame, [board_corners_int], isClosed=True, color=(255, 0, 0), thickness=3)
