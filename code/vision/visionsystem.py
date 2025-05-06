import cv2
import numpy as np

# 내부 모듈 임포트 --------------------------------------------------------------
from vision.camera import camera_open, frame_process
from vision.board import (
    board_detect,
    board_draw,
    board_pts,
    perspective_transform,
    board_origin,
)
from vision.apriltag import AprilTagDetector, cm_per_px
from vision.tracking import TrackingManager
from visual import slider_create, grid_visual, grid_tag_visual, info_tag
from grid import load_grid
from config import (
    tag_info,
    object_points,
    camera_matrix,
    dist_coeffs,
)


class VideoProcessor:
    """종합 영상처리 파이프라인을 하나의 클래스로 통합합니다.

    Responsibilities
    -----------------
    * 카메라 프레임 획득
    * 보드(실험장) 검출 및 원근 변환
    * AprilTag 검출 + 좌표 변환 + 추적
    * 그리드/태그 시각화 프레임 생성
    * 결과를 dict 형태로 반환 (main 등 외부에서 활용)
    """

    # ---------------------------------------------------------------------
    # 초기화
    # ---------------------------------------------------------------------
    def __init__(self, camera_idx: int = 1, tracking_window: int = 5):
        self.cap, self.fps = camera_open()
        self.frame_count = 0

        self.grid_array = load_grid()

        self.tracking_manager = TrackingManager(window_size=tracking_window)
        self.tag_detector = AprilTagDetector()
        slider_create()

    # ---------------------------------------------------------------------
    # 프레임 처리
    # ---------------------------------------------------------------------
    def process_frame(self):
        """하나의 프레임을 처리하고 결과 dict를 반환합니다.

        Returns
        -------
        dict | None
            - None : 프레임 획득 실패
            - dict : 아래 키 포함 (필요에 따라 확장)
              * raw_frame        : 렌즈 왜곡 보정까지 마친 원본 프레임
              * processed_frame  : 태그/보드가 그려진 디스플레이용 프레임
              * warped           : 보드 정사영(리사이즈) 영상
              * grid_visual      : 격자 시각화 이미지
              * tag_info         : config.tag_info의 사본
        """
        # 1. 프레임 읽기 ---------------------------------------------------------------
        self.frame_count += 1
        t_sec = self.frame_count / max(self.fps, 1)
        frame, gray = frame_process(self.cap, camera_matrix, dist_coeffs)
        if frame is None:
            return None

        result = {
            "raw_frame": frame,
            "grid_array": self.grid_array,
        }

        # 2. 보드(실험장) 검출 -----------------------------------------------------------
        largest_rect = board_detect(gray)
        if largest_rect is None:
            # 보드가 안 보이면 최소 정보만 반환
            return result

        board_draw(frame, largest_rect)
        rect, board_w_px, board_h_px = board_pts(largest_rect)
        warped, warped_w_px, warped_h_px, warped_resized = perspective_transform(
            frame, rect, board_w_px, board_h_px
        )
        board_origin_tvec = board_origin(frame, rect[0])

        # 3. AprilTag 검출 -------------------------------------------------------------
        cm_px = cm_per_px(warped_w_px, warped_h_px)
        tags = self.tag_detector.tag_detect(gray)
        self.tag_detector.tags_process(
            tags,
            object_points,
            self.frame_count,
            board_origin_tvec,
            cm_px,
            frame,
            camera_matrix,
            dist_coeffs,
        )

        # 4. 태그 추적(이동 평균/속도) -----------------------------------------------------
        self.tracking_manager.update_all(tag_info, t_sec)

        # 5. 그리드 & 태그 시각화 --------------------------------------------------------
        grid_vis = grid_visual(self.grid_array)
        grid_tag_visual(grid_vis, tag_info)

        # 6. 부가 정보 오버레이 ----------------------------------------------------------
        info_tag(frame, tag_info)

        # 7. 결과 dict 채우기 -----------------------------------------------------------
        result.update(
            {
                "processed_frame": frame,
                "warped": warped_resized,
                "grid_visual": grid_vis,
                "tag_info": tag_info.copy(),
            }
        )
        return result

    # ---------------------------------------------------------------------
    # 자원 해제
    # ---------------------------------------------------------------------
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    # 파이썬 가비지 컬렉션 시 안전장치
    def __del__(self):
        self.release()
