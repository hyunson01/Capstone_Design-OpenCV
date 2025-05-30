import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class BoardDetectionResult:
    corners: np.ndarray
    origin: np.ndarray
    width_px: float
    height_px: float
    cm_per_px: tuple
    perspective_matrix: np.ndarray
    warped: np.ndarray
    warped_resized: np.ndarray
    grid_reference: dict | None = None 
    found: bool = True

class BoardDetector:
    def __init__(self, board_width_cm: float, board_height_cm: float, grid_width: int, grid_height: int):
        self.board_width_cm = board_width_cm
        self.board_height_cm = board_height_cm
        self._result = None
        self._locked = False
        self.grid_width = grid_width
        self.grid_height = grid_height

    def process(self, frame_gray, detect_params=None) -> BoardDetectionResult | None:
        if self._locked:
            return self._result
        
        brightness, min_aspect_ratio, max_aspect_ratio = detect_params
        rect = self._detect_board(frame_gray, brightness, min_aspect_ratio, max_aspect_ratio)

        if rect is not None:
            corners, width_px, height_px = self._get_board_pts(rect)
            origin = self._get_board_origin(corners[0])
            warped, warped_resized, perspective_matrix, warped_width_px, warped_height_px = self._warp_board(frame_gray, corners, width_px, height_px)
            cm_per_px = self._calculate_cm_per_px(warped_width_px, warped_height_px)
            
            self._result = BoardDetectionResult(
                corners=corners,
                origin=origin,
                width_px=width_px,
                height_px=height_px,
                cm_per_px=cm_per_px,
                perspective_matrix=perspective_matrix,
                warped=warped,
                warped_resized=warped_resized,
            )
            return self._result
        else:
            print("[DEBUG] 보드 탐지 실패. 이전 결과 유지됨")
            return self._result
    
    def generate_coordinate_system(self):
        if self._result is None:
            raise RuntimeError("Board not yet detected")

        warped_height, warped_width = self._result.warped.shape[:2]
        cell_width = warped_width / self.grid_width
        cell_height = warped_height / self.grid_height

        cell_centers = []
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                center_x = (col + 0.5) * cell_width
                center_y = (row + 0.5) * cell_height
                cell_centers.append((center_x, center_y))

        # grid lines
        horizontal = []
        for i in range(self.grid_height + 1):
            y = i * cell_height
            horizontal.append(((0, y), (warped_width, y)))

        vertical = []
        for j in range(self.grid_width + 1):
            x = j * cell_width
            vertical.append(((x, 0), (x, warped_height)))

        return {
            "cell_centers": cell_centers,
            "grid_lines": {
                "horizontal": horizontal,
                "vertical": vertical
            }
        }

    def lock(self):
        print("[DEBUG] lock() called")
        if self._result is None:
            print("[ERROR] self._result is None → 탐지된 보드가 없음. 고정 실패")
            return
        self._result.grid_reference = self.generate_coordinate_system()
        print("[DEBUG] grid_reference OK:", self._result.grid_reference is not None)
        self._locked = True
        print("[DEBUG] 보드 고정 완료, locked =", self._locked)

    def reset(self):
        self._locked = False
        self._result = None

    def get_result(self) -> BoardDetectionResult | None:
        return self._result
    
    #!!! 외부에서 결과를 입력 받음. 꼭 이렇게 해야할까?
    def draw(self, frame, result: BoardDetectionResult):
        if result is None:
            return

        # 기본 시각화
        cv2.drawContours(frame, [result.corners.astype(np.int32)], -1, (255, 0, 0), 2)
        cv2.circle(frame, tuple(result.origin[:2].astype(int)), 5, (0, 0, 255), -1)

        if result.grid_reference is None:
            return

        overlay = self.get_grid_overlay_points(result)
        if overlay is None:
            return

        for x, y in overlay["centers"]:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        for pt1, pt2 in overlay["lines"]:
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 255), 1)


    def get_grid_overlay_points(self, result: BoardDetectionResult):
        if result is None or result.grid_reference is None:
            return None

        try:
            inv_matrix = np.linalg.inv(result.perspective_matrix)
        except np.linalg.LinAlgError:
            return None

        # 중심점 변환
        pts = np.array(result.grid_reference["cell_centers"], dtype=np.float32).reshape(-1, 1, 2)
        mapped_centers = cv2.perspectiveTransform(pts, inv_matrix).reshape(-1, 2)

        # 선분 변환
        mapped_lines = []
        grid_lines = result.grid_reference["grid_lines"]
        for segment_list in (grid_lines["horizontal"], grid_lines["vertical"]):
            for p1, p2 in segment_list:
                pts = np.array([[p1, p2]], dtype=np.float32)  # shape (1, 2, 2)
                transformed = cv2.perspectiveTransform(pts, inv_matrix)[0]
                mapped_lines.append((tuple(transformed[0]), tuple(transformed[1])))

        return {
            "centers": mapped_centers,
            "lines": mapped_lines
        }


    # 내부 유틸 함수들
    def _detect_board(self, gray, brightness, min_aspect_ratio, max_aspect_ratio):
        _, thresh = cv2.threshold(gray, brightness, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_rect = None
        largest_area = 0

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if area > 2000 and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                    largest_area = area
                    largest_rect = approx

        return largest_rect

    def _get_board_pts(self, rect):
            pts = rect.reshape(4, 2).astype(np.float32)
            sum_pts = pts.sum(axis=1)
            diff_pts = np.diff(pts, axis=1)
            top_left = pts[np.argmin(sum_pts)]
            bottom_right = pts[np.argmax(sum_pts)]
            top_right = pts[np.argmin(diff_pts)]
            bottom_left = pts[np.argmax(diff_pts)]

            ordered = np.array([top_left, top_right, bottom_right, bottom_left])
            width_px = np.linalg.norm(top_right - top_left)
            height_px = np.linalg.norm(bottom_left - top_left)

            return ordered, width_px, height_px

    def _calculate_cm_per_px(self, warped_width_px, warped_height_px):
        cm_per_px_x = self.board_width_cm / max(warped_width_px, 1)
        cm_per_px_y = self.board_height_cm / max(warped_height_px, 1)
        return (cm_per_px_x, cm_per_px_y)

    def _warp_board(self, frame, corners, board_width_px, board_height_px):
        dst = np.array([[0, 0], [board_width_px - 1, 0],
                    [board_width_px - 1, board_height_px - 1], [0, board_height_px - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(frame, matrix, (int(board_width_px), int(board_height_px)))
        warped_resized = cv2.resize(warped, (frame.shape[1] // 2, frame.shape[1] // 2))
        warped_board_width_px = int(np.linalg.norm(dst[1] - dst[0]))
        warped_board_height_px = int(np.linalg.norm(dst[3] - dst[0]))
        return warped, warped_resized, matrix, warped_board_width_px, warped_board_height_px
    
    def _get_board_origin(self, top_left):
        return np.array([top_left[0], top_left[1], 0], dtype=np.float32)
    
    @property
    def is_locked(self):
        return self._locked
