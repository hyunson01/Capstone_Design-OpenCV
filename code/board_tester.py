import cv2
import numpy as np
from vision.camera import camera_open, Undistorter  # fileciteturn0file2
from vision.visionsystem import ROIFilter         # fileciteturn0file1
from vision.board import BoardDetector           # fileciteturn0file0
from config import camera_cfg, board_width_cm, board_height_cm, grid_row, grid_col

# --- 전역 변수: ROI 선택 상태 ---
roi_tl = None       # Top-left 좌표
roi_br = None       # Bottom-right 좌표
selecting_roi = False

# 마우스 콜백: 좌클릭 두 번으로 ROI 설정
def mouse_callback(event, x, y, flags, param):
    global roi_tl, roi_br, selecting_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        if not selecting_roi:
            roi_tl = (x, y)
            selecting_roi = True
            print(f"[ROI] Top-left set to: {roi_tl}")
        else:
            roi_br = (x, y)
            selecting_roi = False
            print(f"[ROI] Bottom-right set to: {roi_br}")


def main():
    global roi_tl, roi_br, selecting_roi
    # 1) 카메라 열기 및 왜곡 보정기 설정
    cap, fps = camera_open()  # 기본 카메라 검색 및 설정 fileciteturn0file2
    undistorter = Undistorter(
        camera_cfg['type'], camera_cfg['matrix'], camera_cfg['dist'], camera_cfg['size']
    )

    # 2) 보드 탐지기 및 필터 초기화
    board_detector = BoardDetector(
        board_width_cm, board_height_cm, grid_row, grid_col
    )  # 보드 크기(cm)와 그리드 정보 설정 fileciteturn0file0
    roi_filter = ROIFilter()      # 전처리용 필터 fileciteturn0file1

    # 3) 윈도우 및 콜백 등록
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback)

    # 탐지 파라미터: (min_aspect_ratio, max_aspect_ratio)
    detect_params = (0, 0.5, 2.0, 0.5, 0.5, 0.6)  # 초기값 설정

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임 획득 실패")
            break

        # 왜곡 보정 및 회색조
        undistorted, _ = undistorter.undistort(frame)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        display = undistorted.copy()

        # 4) ROI가 설정되었으면 그 영역에서 전처리 및 보드 탐지
        if roi_tl and roi_br:
            x1, y1 = roi_tl
            x2, y2 = roi_br
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            if x2 > x1 and y2 > y1:
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    print("[ERROR] Empty ROI region")
                else:
                    # 이진화 전처리 및 필터 결과 보기
                    filtered = roi_filter.binarize(roi)  # fileciteturn0file1
                    cv2.imshow("Filtered ROI", filtered)

                    # 보드 검출
                    rect = board_detector.detect(filtered, detect_params)  # fileciteturn0file0
                    if rect is not None:
                        rect_full = rect.copy()
                        rect_full[:,0,0] += x1
                        rect_full[:,0,1] += y1
                        cv2.drawContours(display, [rect_full.astype(np.int32)], -1, (0,255,0), 2)

                        # 퍼스펙티브 워프 예시 (선택)
                        corners, w_px, h_px = board_detector._get_board_pts(rect_full)
                        dst = np.array([[0,0], [w_px-1,0], [w_px-1,h_px-1], [0,h_px-1]], dtype=np.float32)
                        H = cv2.getPerspectiveTransform(corners, dst)
                        warped = cv2.warpPerspective(gray, H, (int(w_px), int(h_px)))
                        cv2.imshow("Warped Board", warped)
            else:
                print(f"[ERROR] Invalid ROI coordinates: tl={roi_tl}, br={roi_br}")

        # ROI 영역 시각화
        if roi_tl:
            cv2.circle(display, roi_tl, 5, (255,0,0), -1)
        if roi_tl and roi_br:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0,0,255), 2)

        cv2.imshow("Frame", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            roi_tl, roi_br = None, None
            selecting_roi = False
            print("[INFO] ROI reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
