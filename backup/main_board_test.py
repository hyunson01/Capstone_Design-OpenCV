# main.py
import sys
import os
import cv2
import numpy as np

sys.path.append(r"C:\Users\2001t\Downloads\Capstone_Design-OpenCV-main_0506\Capstone_Design-OpenCV-main\MAPF-ICBS\code")

from vision.camera import camera_open, frame_process
from vision.apriltag import AprilTagDetector
from vision.tracking import TrackingManager
from vision.new_board import get_board_corners_from_tag, draw_board_outline
from visual import mouse_callback, grid_visual, grid_tag_visual, info_tag, slider_create
from config import tag_info, object_points, camera_matrix, dist_coeffs, cbs_path, arguments
from grid import save_grid, load_grid
##from cbs.cbs_runner import run_cbs_manager

# ====== 주요 설정 ======
APRILTAG_LENGTH_CM = 5.0
BOARD_WIDTH_CM = 120.0
BOARD_HEIGHT_CM = 90.0

def main():
    cap, fps = camera_open()
    frame_count = 0
    tag_detector = AprilTagDetector()
    tracking_manager = TrackingManager(window_size=5)
    grid_array = load_grid()

    slider_create()
    cv2.namedWindow("Grid Visualization")
    cv2.setMouseCallback("Grid Visualization", mouse_callback, param=grid_array)

    while True:
        frame_count += 1
        time = frame_count / fps
        frame, gray = frame_process(cap, camera_matrix, dist_coeffs)
        if frame is None:
            continue

        # === 태그 인식 ===
        tags = tag_detector.tag_detect(gray)
        if not tags:
            cv2.imshow("AprilTag Board", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            continue

        # === 태그 처리 ===
        cm_per_px = (1.0, 1.0)  # 튜플로 수정

        tag_detector.tags_process(
            tags,
            object_points,
            frame_count,
            origin_px=(0, 0),
            cm_per_px=cm_per_px,
            frame=frame,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs
        )
        tracking_manager.update_all(tag_info, time)

        # === 기준 태그 (왼쪽 아래) 가져오기 ===
        base_tag_id = list(tag_info.keys())[0]
        base_tag_data = tag_info[base_tag_id]

        if "rvec" not in base_tag_data or "tvec" not in base_tag_data:
            continue

        # === 경기장 윤곽선 생성 ===
        board_corners = get_board_corners_from_tag(
            base_tag_data["rvec"],
            base_tag_data["tvec"],
            APRILTAG_LENGTH_CM,
            BOARD_WIDTH_CM,
            BOARD_HEIGHT_CM,
            camera_matrix,
            dist_coeffs
        )
        draw_board_outline(frame, board_corners)

        # === 정보 시각화 ===
        info_tag(frame, tag_info)
        grid_vis = grid_visual(grid_array)
        grid_tag_visual(grid_vis, tag_info)

        # === 디스플레이 ===
        cv2.imshow("AprilTag Board", frame)
        cv2.imshow("Grid Visualization", grid_vis)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_grid(grid_array)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
