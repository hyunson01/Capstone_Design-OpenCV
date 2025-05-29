import cv2
import numpy as np
from vision.board import BoardDetector
from vision.apriltag import AprilTagDetector
from config import board_width_cm, board_height_cm, grid_row, grid_col, cell_size, camera_matrix, dist_coeffs
from vision.camera import camera_undistort

class VisionSystem:
    def __init__(self, visualize=True):
        self.board = BoardDetector(board_width_cm, board_height_cm, grid_row, grid_col)
        self.tags = AprilTagDetector()
        self.visualize = visualize
        self.last_valid_result = None
        self.frame_count = 0

    # 프레임 처리
    def process_frame(self, raw_frame, detect_params=None):

        frame, new_camera_matrix = camera_undistort(raw_frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        # 1. 보드 탐지 또는 고정 상태 유지
        self.board.process(gray, detect_params)
        board_result = self.board.get_result()
                
        # 2. 태그 탐지 및 처리
        tags = self.tags.detect(gray)

        if board_result:
            self.tags.update_and_process(tags, self.frame_count, board_result.origin, board_result.cm_per_px, new_camera_matrix)
        else:
            self.tags.update(tags, self.frame_count, new_camera_matrix)

        tag_info = self.tags.get_raw_tags()
        if board_result:
            self.transform_coordinates(tag_info)

        # 3. 시각화 처리
        if self.visualize:
            self.board.draw(frame, board_result)
            self.tags.draw(frame)
            self.draw_tag_overlay(frame, tag_info)

        return {
            "frame": frame,
            "tag_info": tag_info,
        }

    # 외부 이벤트 처리용 API들
    def lock_board(self):
        self.board.lock()

    def reset_board(self):
        self.board.reset()
        self.last_valid_result = None

    def toggle_visualization(self):
        self.visualize = not self.visualize

    def get_raw_tag_info(self):
        return self.tags.get_raw_tags()

    def get_robot_tags(self):
        return self.tags.get_robot_tags()
    
    def get_fps(self):
        return self.fps

    # 각 태그의 좌표를 그리드 좌표로 변환
    def transform_coordinates(self, tag_info: dict[int, dict]) -> dict[int, dict]:
        result = {}
        cell_w = board_width_cm / grid_col
        cell_h = board_height_cm / grid_row
        for tag_id, data in tag_info.items():
            x_cm, y_cm = data["coordinates"]
            col = int(x_cm / cell_w)
            row = int(y_cm / cell_h)

            row = max(0, min(grid_row - 1, row))
            col = max(0, min(grid_col - 1, col))
            data["grid_position"] = (row, col)
    
    def draw_tag_overlay(self, frame, tag_info):
        base_x = 10
        base_y = 30
        line_height = 20

        for idx, (tag_id, data) in enumerate(sorted(tag_info.items())):
            if data["status"] != "On":
                continue
            v = data.get("velocity", (0, 0))
            vx, vy = v
            text = f"ID {tag_id}: V=({vx:.2f}, {vy:.2f})"
            pos = (base_x, base_y + idx * line_height)
            # cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            smoothed = data.get("smoothed_coordinates")
            if smoothed:
                sx, sy = int(smoothed[0]), int(smoothed[1])
                ex, ey = int(sx + vx * 50), int(sy + vy * 50)
                # cv2.arrowedLine(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)
