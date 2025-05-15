import cv2
import numpy as np
import glob
import os

# 체커보드 크기 설정 (내부 코너 개수)
CHECKERBOARD = (10,7)  # 내부 코너 개수 (체커보드 패턴에 맞게 조정)
square_size = 0.025  # 체커보드 칸 크기 (미터 단위, 실제 크기에 맞춰 조정)

# 체커보드 찾기 알고리즘의 종료 기준 설정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D 세계 좌표 준비 (Z=0 평면 상의 3D 점)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# 3D 점과 2D 점 저장할 리스트
objpoints = []  # 3D 공간 좌표
imgpoints = []  # 2D 이미지 좌표

# 현재 실행 중인 Python 파일의 경로 찾기
script_dir = os.path.dirname(os.path.abspath(__file__))
images_path = r"C:\img\calibration\*.jpg"
images = glob.glob(images_path)

# 이미지가 없을 경우 프로그램 종료
if not images:
    print("⚠️ No images found in the 'images' directory!")
    exit()

valid_images = 0  # 정상적으로 처리된 이미지 개수

for fname in images:
    print(f"Trying to read: {fname}")  # 디버깅을 위한 이미지 경로 출력
    img = cv2.imread(fname)

    if img is None:
        print(f"❌ Error: Could not load image {fname}")  # 이미지 불러오기 실패 시 경고
        continue  # 이미지 로드 실패 시 다음 이미지로 건너뛰기

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백 변환

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # 코너 미세 조정
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # 체커보드 코너 그리기
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibration Image', img)
        cv2.waitKey(500)

        valid_images += 1  # 유효한 이미지 개수 증가

cv2.destroyAllWindows()

# 📌 유효한 이미지가 하나도 없으면 종료
if valid_images == 0:
    print("❌ No valid images for calibration. Please check your images.")
    exit()

# 카메라 캘리브레이션 수행
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과 출력
print("✅ Camera Matrix:\n", cameraMatrix)
print("✅ Distortion Coefficients:\n", distCoeffs)

# 📌 보정값 저장
# 이미지들이 있는 폴더 경로 가져오기
image_folder = os.path.dirname(images[0])  # 첫 번째 이미지 기준으로 경로 추출

# 저장 경로 구성
camera_matrix_path = os.path.join(image_folder, "camera_matrix.npy")
dist_coeffs_path = os.path.join(image_folder, "dist_coeffs.npy")

# 저장
np.save(camera_matrix_path, cameraMatrix)
np.save(dist_coeffs_path, distCoeffs)

print(f"✅ Calibration data saved to:\n{camera_matrix_path}\n{dist_coeffs_path}")
