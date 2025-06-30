import cv2
import numpy as np
import math
from vision.apriltag import AprilTagDetector
from config import board_width_cm, board_height_cm, grid_row, grid_col, cell_size
from vision.board import BoardDetectionResult

class VisionSystem:
    def __init__(self, undistorter, visualize=True, board_mode='tag'):
        
        self.tags = AprilTagDetector()
        self.visualize = visualize
        self.grid_row = grid_row
        self.grid_col = grid_col
        self.board_mode = board_mode
        self.undistorter = undistorter
        self.last_valid_result = None
        self.board_result: BoardDetectionResult | None = None
        self.frame_count = 0
        self._init_board_detector()

    def _init_board_detector(self):
        if self.board_mode == 'contour':
            from vision.board import BoardDetector
            self.board = BoardDetector(board_width_cm, board_height_cm, grid_row, grid_col)
        elif self.board_mode == 'tag':
            from vision.board_tag import TagBoardDetector
            self.board = TagBoardDetector(board_width_cm, board_height_cm, grid_row, grid_col)


    def set_board_mode(self, new_mode):
        self.board_mode = new_mode
        self._init_board_detector()
        
    # 프레임 처리
    def process_frame(self, raw_frame, detect_params=None):

        frame, new_camera_matrix = self.undistorter.undistort(raw_frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        tags = self.tags.detect(gray)
        
        # 1. 보드 탐지 또는 고정 상태 유지
        if self.board_mode == 'tag':
            data = tags
        else:
            data = detect_params

        self.board.process(gray, data)
        self.board_result = self.board.get_result()
                
        # 2. 태그 탐지 및 처리
        if self.board_result:
            self.tags.update_and_process(tags, self.frame_count, self.board_result.origin, self.board_result.cm_per_px, new_camera_matrix)
        else:
            self.tags.update(tags, self.frame_count, new_camera_matrix)

        tag_info = self.tags.get_raw_tags()
        if self.board_result:
            self.transform_coordinates(tag_info)

        # 3. 시각화 처리
        if self.visualize:
            self.board.draw(frame, self.board_result)
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

    def transform_coordinates(self, tag_infos: dict[int, dict]):
        """
        각 태그의 원본 이미지 좌표를 bird's-eye 뷰로 투영하여 그리드 인덱스를 계산하고,
        보드가 잠긴 상태일 때만 그리드 중심까지의 거리를 cm 단위로 계산해 저장합니다.
        """
        # 1) 아직 보드가 감지되지 않았으면 중단
        if not self.board_result:
            return

        # 2) 호모그래피 행렬과 warped 뷰 크기
        M = self.board_result.perspective_matrix
        warped_h, warped_w = self.board_result.warped.shape[:2]

        # 3) 그리드 셀 크기 (bird’s-eye 기준)
        cell_w = warped_w / self.grid_col
        cell_h = warped_h / self.grid_row

        # 4) 보드가 잠긴 경우에만 grid_reference에서 중심점 리스트 가져오기
        if self.board.is_locked and self.board_result.grid_reference:
            grid_centers = self.board_result.grid_reference["cell_centers"]
            sx, sy = self.board_result.cm_per_px
            scale = math.sqrt(sx**2 + sy**2)
        else:
            grid_centers = None
            scale = None

        # 5) 태그별 처리
        for tag_id, data in tag_infos.items():
            cx_px, cy_px = data.get("center", (None, None))
            if cx_px is None or cy_px is None:
                continue

            # bird’s-eye 뷰로 투영
            pts = np.array([[[cx_px, cy_px]]], dtype=np.float32)
            x_w, y_w = cv2.perspectiveTransform(pts, M)[0][0]

            # grid 인덱스 계산
            col = int(x_w // cell_w)
            row = int(y_w // cell_h)
            col = max(0, min(self.grid_col - 1, col))
            row = max(0, min(self.grid_row - 1, row))
            data["grid_position"] = (row, col)

            # 보드가 잠긴 상태에서만 거리 계산
            if grid_centers is not None:
                idx = row * self.grid_col + col
                if idx < len(grid_centers):
                    center_x, center_y = grid_centers[idx]
                    dist_px = ((x_w - center_x)**2 + (y_w - center_y)**2)**0.5
                    data["dist_cm"] = dist_px * scale
                else:
                    data["dist_cm"] = None
            else:
                data["dist_cm"] = None




    
    def draw_tag_overlay(self, frame, tag_info):
        """
        화면에 각 태그의 ID와 bird’s-eye 기반 거리(dist_cm)를 cm 단위로 표시합니다.
        보드가 lock 되기 전에는 dist_cm이 None이므로 “–”로 표시됩니다.
        """
        base_x = 10
        base_y = 30
        line_height = 20

        for idx, (tag_id, data) in enumerate(sorted(tag_info.items())):
            # 태그가 Off 상태면 건너뜀
            if data.get("status") != "On":
                continue

            # 거리 정보 가져오기 (None 가능)
            dist = data.get("dist_cm")
            if dist is None:
                text = f"ID {tag_id}: d=–"
            else:
                text = f"ID {tag_id}: d={dist:.2f}cm"

            # 텍스트 위치 계산 및 그리기
            pos = (base_x, base_y + idx * line_height)
            cv2.putText(
                frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1
            )
