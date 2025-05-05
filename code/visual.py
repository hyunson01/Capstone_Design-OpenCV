import cv2
import numpy as np

# 상수 지정
board_height_cm = 30
board_width_cm = 30
grid_width = 600    # 12*50
grid_height = 600
cell_size = 50

def grid_ini(rows=12, cols=12):
    return np.zeros((rows, cols), dtype=int)

def grid_tag_visual(grid_visual, tag_info):
    pass

def info_tag(frame, tag_info):
    pass

def trackbar(val):
    pass

def slider_create():
    cv2.namedWindow("Detected Rectangle")
    cv2.createTrackbar("Brightness Threshold", "Detected Rectangle", 120, 255, trackbar)
    cv2.createTrackbar("Min Aspect Ratio", "Detected Rectangle", 10, 20, trackbar)
    cv2.createTrackbar("Max Aspect Ratio", "Detected Rectangle", 15, 20, trackbar)

def slider_value():
    brightness_threshold = cv2.getTrackbarPos("Brightness Threshold", "Detected Rectangle")
    min_aspect_ratio = cv2.getTrackbarPos("Min Aspect Ratio", "Detected Rectangle") / 10.0
    max_aspect_ratio = cv2.getTrackbarPos("Max Aspect Ratio", "Detected Rectangle") / 10.0
    return brightness_threshold, min_aspect_ratio, max_aspect_ratio

def grid_visual(grid_array):
    visual = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    for i in range(grid_array.shape[0]):
        for j in range(grid_array.shape[1]):
            cell_x = j * cell_size
            cell_y = i * cell_size
            color = (0, 0, 0) if grid_array[i, j] == 1 else (255, 255, 255)
            cv2.rectangle(visual, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), color, -1)

    # 격자선 그리기
    for i in range(grid_array.shape[0] + 1):
        cv2.line(visual, (0, i * cell_size), (grid_width, i * cell_size), (200, 200, 200), 1)

    for j in range(grid_array.shape[1] + 1):
        cv2.line(visual, (j * cell_size, 0), (j * cell_size, grid_height), (200, 200, 200), 1)

    return visual

# 마우스 상태 변수
is_mouse_pressed = False
last_toggled = None

# mouse_callback(grid_array)를 받게 수정 ✅
def mouse_callback(event, x, y, flags, param):
    global is_mouse_pressed, last_toggled

    grid_array = param   # 여기서 param으로 grid_array 받아쓰기
    row, col = y // cell_size, x // cell_size
    if not (0 <= row < grid_array.shape[0] and 0 <= col < grid_array.shape[1]):
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        is_mouse_pressed = True
        grid_array[row, col] = 1 - grid_array[row, col]
        last_toggled = (row, col)
        cv2.imshow("Grid", grid_visual(grid_array))

    elif event == cv2.EVENT_MOUSEMOVE and is_mouse_pressed:
        if last_toggled != (row, col):
            grid_array[row, col] = 1 - grid_array[row, col]
            last_toggled = (row, col)
            cv2.imshow("Grid", grid_visual(grid_array))

    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_pressed = False
        last_toggled = None
