import cv2
import numpy as np
from pupil_apriltags import Detector

#보드 크기
board_width_cm = 25.7  # 보드의 너비 (예: 60cm)
board_height_cm = 18.2  # 보드의 높이 (예: 50cm)

#카메라 매개변수 (캘리브레이션된 값 사용)
camera_matrix = np.array([[1.44881455e+03,0,9.80488323e+02],[0,1.45151609e+03,5.39528675e+02],[0,0,1.00000000e+00]], dtype=np.float32)  # 예제 값 (캘리브레이션 필요)
dist_coeffs = np.array([[ 3.75921025e-03,1.02703292e-01,-7.06313415e-05,1.59368677e-03,-2.21477882e-01]])  # 예제 왜곡 계수 (캘리브레이션 필요)

#그리드창 크기
grid_x=600
grid_y=int(grid_x * board_height_cm / board_width_cm)

# 3D 상자 정의
box_size = 0.05  # 태그 크기 (단위: 미터)
object_points = np.array([
    [0, 0, 0], [box_size, 0, 0], [box_size, box_size, 0], [0, box_size, 0],  # 태그 평면
    [0, 0, box_size], [box_size, 0, box_size], [box_size, box_size, box_size], [0, box_size, box_size]  # 상단 꼭짓점
], dtype=np.float32)

# AprilTag 탐지기 초기화
detector = Detector(families="tag36h11")

# 태그 상태 관리
tag_info = {}  # 태그 ID 매핑 {태그 ID: (상태, 좌표, 기울기, 마지막 감지된 프레임)}

# 카메라 설정 < 모듈화 완완
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()
    
# 슬라이더 콜백 함수
def on_trackbar(val):
    pass

# 윈도우 생성 및 슬라이더 추가
cv2.namedWindow("Detected Rectangle")
cv2.createTrackbar("Brightness Threshold", "Detected Rectangle", 120, 255, on_trackbar)
cv2.createTrackbar("Min Aspect Ratio", "Detected Rectangle", 12, 20, on_trackbar)  # 기본값 1.2
cv2.createTrackbar("Max Aspect Ratio", "Detected Rectangle", 15, 20, on_trackbar)  # 기본값 1.5

# 격자 배열 생성
grid_array = np.zeros((5, 6), dtype=int)  # 5행 x 6열의 배열 (가로 6, 세로 5)

frame_count = 0  # 현재 프레임 번호

while True:
    
    # <모듈
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera.")
        break

    frame_count += 1

    #카메라 왜곡 보정 < 모듈
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 흑백 변환 < 모듈
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
    # 슬라이더 값 가져오기
    brightness_threshold = cv2.getTrackbarPos("Brightness Threshold", "Detected Rectangle")
    min_aspect_ratio = cv2.getTrackbarPos("Min Aspect Ratio", "Detected Rectangle") / 10.0
    max_aspect_ratio = cv2.getTrackbarPos("Max Aspect Ratio", "Detected Rectangle") / 10.0

    # 2. 밝기 기준으로 이진화 < 모듈듈
    _, thresh = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    # 3. Contours 찾기 < 모듈듈
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 직사각형 초기값 < 모듈듈
    largest_area = 0
    largest_rect = None

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            min_aspect_ratio = cv2.getTrackbarPos("Min Aspect Ratio", "Detected Rectangle") / 10.0
            max_aspect_ratio = cv2.getTrackbarPos("Max Aspect Ratio", "Detected Rectangle") / 10.0

            if area > 5000 and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                largest_area = area
                largest_rect = approx

    if largest_rect is not None:
        cv2.drawContours(frame, [largest_rect], -1, (255, 0, 0), 3)

        pts = largest_rect.reshape(4, 2).astype(np.float32)

        # 정렬
        sum = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        top_left = pts[np.argmin(sum)]
        bottom_right = pts[np.argmax(sum)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]

        # 보드 크기 측정 (픽셀 단위)
        board_width_px = np.linalg.norm(top_right - top_left)
        board_height_px = np.linalg.norm(bottom_left - top_left)

        # Perspective Transformation
        rect = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
        dst = np.array([[0, 0], [board_width_px - 1, 0], 
                        [board_width_px - 1, board_height_px - 1], [0, board_height_px - 1]], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, matrix, (int(board_width_px), int(board_height_px)))
        
        # 변환 후 보드 크기 측정
        warped_board_width_px = int(np.linalg.norm(dst[1] - dst[0]))  # (top_right - top_left)
        warped_board_height_px = int(np.linalg.norm(dst[3] - dst[0])) # (bottom_left - top_left)
        
        warped_resized = cv2.resize(warped, (frame.shape[1] // 2, frame.shape[0] // 2))
        board_origin_tvec = np.array([top_left[0], top_left[1], 0], dtype=np.float32)
        
        cv2.circle(frame, tuple(board_origin_tvec[:2].astype(int)), 5, (0, 0, 255), -1)
        
        # 픽셀당 cm 변환 비율 (0으로 나누는 것 방지)
        cm_per_px_x = board_width_cm / max(warped_board_width_px, 1)
        cm_per_px_y = board_height_cm / max(warped_board_height_px, 1)

        # 그리드 관련
        grid_visual = np.ones((grid_y, grid_x, 3), dtype=np.uint8) * 255  # 흰색 배경
        for i in range(5):
            for j in range(6):
                cell_x = j * 100
                cell_y = i * 100
                color = (0, 255, 0) if grid_array[i, j] == 1 else (255, 255, 255)
                cv2.rectangle(grid_visual, (cell_x, cell_y), (cell_x + 100, cell_y + 100), (0, 0, 0), 1)
                cv2.rectangle(grid_visual, (cell_x, cell_y), (cell_x + 100, cell_y + 100), color, -1)
                
        # AprilTag 탐지
        tags = detector.detect(gray)
        detected_ids = set()  # 현재 프레임에서 탐지된 태그 ID

        # 감지되지 않은 태그 업데이트
        for tag_id in list(tag_info.keys()):
            if tag_id not in detected_ids:
                last_seen_frame = tag_info[tag_id][4]
                if frame_count - last_seen_frame > 30:  # 약 1초 동안 감지되지 않으면
                    tag_info[tag_id] = ("Off", tag_info[tag_id][1], tag_info[tag_id][2],tag_info[tag_id][3], last_seen_frame)
        
        #태그 관련
        for tag in tags:
            tag_id = tag.tag_id
            corners = tag.corners.astype(np.float32)

            # PnP 알고리즘으로 태그의 3D 자세 추정
            retval, rvec, tvec = cv2.solvePnP(object_points[:4], corners, new_camera_matrix, dist_coeffs)
            
            center = tuple(map(int, tag.center))

            # 회전 벡터 -> Roll, Pitch, Yaw 계산
            rmat, _ = cv2.Rodrigues(rvec)
            yaw, pitch, roll = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvec)))[-1]

            # 태그 좌표
            tag_x = (center[0] - board_origin_tvec[0])
            tag_y = (center[1] - board_origin_tvec[1])
            
            # 태그 좌표를 보드 좌표계로 변환
            tag_cm_x = tag_x * cm_per_px_x
            tag_cm_y = tag_y * cm_per_px_y

            # 태그 상태 업데이트 (좌표와 기울기 포함)
            tag_info[tag_id] = (
                "On",
                (tvec[0][0], tvec[1][0], tvec[2][0]),  # 3D 좌표
                (roll[0], pitch[0], yaw[0]),  # 기울기
                (tag_cm_x, tag_cm_y), # 좌표
                frame_count
            )
            detected_ids.add(tag_id)
            
             # 상자 3D 좌표를 이미지로 투영
            imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, new_camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # 상자 그리기
            cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)  # 바닥
            # cv2.drawContours(frame, [imgpts[4:]], -1, (255, 0, 0), 2)  # 상단
            # for i in range(4):
            #     cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 255, 255), 2)  # 연결선

            # 태그 위 ID 표시
            cv2.putText(frame, f"ID: {tag_id}", (center[0], center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

    
        # 태그 정보 표시
        y_offset = 30
        for idx, (tag_id, (status, coordinates_3d, angles, coordinates, _)) in enumerate(sorted(tag_info.items())):
            id_text = f"ID {tag_id}"
            # coord_text = f"Coord: ({coordinates[0]:.2f}, {coordinates[1]:.2f}, {coordinates[2]:.2f})"
            # angles_text = f"Angles: R({angles[0]:.1f}), P({angles[1]:.1f}), Y({angles[2]:.1f})"
            
            # 좌표 및 기울기 정보를 한 줄씩 출력
            color = (0, 255, 0) if status == "On" else (0, 0, 255)
            cv2.putText(frame, id_text, (10, y_offset + idx * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"({coordinates[0]:.1f}cm, {coordinates[1]:.1f}cm)",(10, y_offset + idx * 60 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # cv2.putText(frame, coord_text, (10, y_offset + idx * 60 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # cv2.putText(frame, angles_text, (10, y_offset + idx * 60 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            tag_grid_x = int(coordinates[0] * grid_x / board_width_cm)
            tag_grid_y = int(coordinates[1] * grid_y / board_height_cm)
            
            # 태그를 그리드에 표시
            if 0 <= tag_grid_x < grid_x and 0 <= tag_grid_y < grid_y:
                cv2.circle(grid_visual, (tag_grid_x, tag_grid_y), 5, (0, 255, 0), -1)  # 초록색 점 찍기
            
        
        # 변환된 화면 및 배열 시각화 표시
        cv2.imshow("Warped Perspective", warped_resized)
        cv2.imshow("Grid Visualization", grid_visual)

    # 디스플레이 (실시간 출력)
    cv2.imshow("Detected Rectangle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()