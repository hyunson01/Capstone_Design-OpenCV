import cv2

def camera_open():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        raise Exception("Error: Cannot open camera.")
    return cap, fps

def camera_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def camera_undistort(frame, camera_matrix, dist_coeffs):
    h, w = frame.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return frame

def frame_process(cap, camera_matrix, dist_coeffs):
    frame = camera_frame(cap)
    if frame is None:
        return None, None
    frame_undistort = camera_undistort(frame, camera_matrix, dist_coeffs)
    frame_gray = cv2.cvtColor(frame_undistort, cv2.COLOR_BGR2GRAY)
    return frame_undistort, frame_gray