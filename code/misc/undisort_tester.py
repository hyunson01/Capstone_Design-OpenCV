import cv2
import numpy as np
from vision.camera import camera_open, Undistorter
from config import camera_cfg

# 카메라 열기
cap, fps = camera_open(source=None)

# 전역 언디스토션 설정
undist = Undistorter(
    camera_cfg['type'],
    camera_cfg['matrix'],
    camera_cfg['dist'],
    camera_cfg['size']
)
map1_glob, map2_glob = undist.map1, undist.map2
camera_matrix = camera_cfg['matrix']
orig_dist = camera_cfg['dist']
image_size = camera_cfg['size']  # (height, width)

# 1차 ROI 변수
roi1_tl = None
roi1_br = None
selecting_roi1 = False
show_roi1 = False

# 2차 ROI 변수
roi2_tl = None
roi2_br = None
selecting_roi2 = False
show_roi2 = False

# ROI 전용 undistorter 초기값
k1_roi = 0.0
new_dist = np.zeros_like(orig_dist)
roi_undist = Undistorter(
    camera_cfg['type'],
    camera_matrix,
    new_dist,
    image_size
)
map1_roi, map2_roi = roi_undist.map1, roi_undist.map2

# 트랙바 콜백: ROI1 내부 k1 조정
def on_trackbar(val):
    global k1_roi, roi_undist, map1_roi, map2_roi
    k1_roi = (val - 50) / 1000.0  # -0.05 ~ +0.05
    dist = np.zeros_like(orig_dist)
    dist[0] = k1_roi
    roi_undist = Undistorter(
        camera_cfg['type'], camera_matrix, dist, image_size
    )
    map1_roi, map2_roi = roi_undist.map1, roi_undist.map2

# 마우스 콜백: 1차 및 2차 ROI 선택
def mouse_cb(event, x, y, flags, param):
    global roi1_tl, roi1_br, selecting_roi1, show_roi1
    global roi2_tl, roi2_br, selecting_roi2, show_roi2
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    # 1차 ROI
    if selecting_roi1:
        if roi1_tl is None:
            roi1_tl = (x, y)
            print(f"[ROI1] Top-left: {roi1_tl}")
        else:
            roi1_br = (x, y)
            selecting_roi1 = False
            show_roi1 = True
            print(f"[ROI1] Bottom-right: {roi1_br}")
    # 2차 ROI (Corrected 창에서만 선택)
    elif selecting_roi2:
        if roi2_tl is None:
            roi2_tl = (x, y)
            print(f"[ROI2] Top-left: {roi2_tl}")
        else:
            roi2_br = (x, y)
            selecting_roi2 = False
            show_roi2 = True
            print(f"[ROI2] Bottom-right: {roi2_br}")

# 창 및 트랙바 설정
cv2.namedWindow('Original')
cv2.namedWindow('Corrected')
cv2.namedWindow('NestedROI')
cv2.setMouseCallback('Corrected', mouse_cb)
cv2.createTrackbar('k1×1000', 'Corrected', 50, 100, on_trackbar)

print("Press 's' to start first ROI, 'n' for nested ROI, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 원본 프레임 표시
    cv2.imshow('Original', frame)

    # 전역 + ROI1 보정
    if show_roi1:
        # 맵 복사 및 1차 ROI 맵 적용
        m1 = map1_glob.copy()
        m2 = map2_glob.copy()
        x0, y0 = roi1_tl
        x1, y1 = roi1_br
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        m1[y_min:y_max, x_min:x_max] = map1_roi[y_min:y_max, x_min:x_max]
        m2[y_min:y_max, x_min:x_max] = map2_roi[y_min:y_max, x_min:x_max]
        corrected = cv2.remap(frame, m1, m2, interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(corrected, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    else:
        corrected = cv2.remap(frame, map1_glob, map2_glob, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('Corrected', corrected)

    # 2차 ROI 창 표시
    if show_roi2 and show_roi1:
        # 1차 보정된 영상에서 2차 ROI 영역 크롭
        x0, y0 = roi2_tl
        x1, y1 = roi2_br
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        nested = corrected[y_min:y_max, x_min:x_max]
        cv2.imshow('NestedROI', nested)
    else:
        # 빈 창 표시
        blank = np.zeros((200,200,3), dtype=np.uint8)
        cv2.imshow('NestedROI', blank)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 1차 ROI 리셋 및 선택 모드
        roi1_tl = roi1_br = None
        show_roi1 = False
        selecting_roi1 = True
        # 중첩 ROI 초기화
        roi2_tl = roi2_br = None
        show_roi2 = False
        selecting_roi2 = False
        print("Enter first ROI: click TL then BR in 'Corrected'.")
    elif key == ord('n') and show_roi1:
        # 2차 ROI 선택 모드
        roi2_tl = roi2_br = None
        show_roi2 = False
        selecting_roi2 = True
        print("Enter nested ROI: click TL then BR in 'Corrected'.")

cap.release()
cv2.destroyAllWindows()
