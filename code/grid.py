import cv2
import numpy as np
import os
import json
from visual import mouse_callback, grid_visual
from config import grid_size, grid_path

# JSON에서 배열 불러오기
def load_grid():
    if os.path.exists(grid_path):
        with open(grid_path, "r") as f:
            data = json.load(f)
            return np.array(data["grid"], dtype=int)  # JSON의 grid 부분을 로드
    else:
        return (np.zeros((grid_size, grid_size), dtype=int))  # 파일 없으면 0으로 초기화

# JSON으로 배열 저장
def save_grid(grid):
    data = {
        "grid": grid.tolist()  # numpy 배열을 리스트로 변환하여 저장
    }
    with open(grid_path, "w") as f:
        json.dump(data, f, indent=4)  # 보기 좋게 저장
    print(f"저장 완료: {grid_path}")

# grid.py의 마지막에 추가
def main():
    grid_array = load_grid()

    cv2.namedWindow("Grid")
    cv2.setMouseCallback("Grid", mouse_callback, param=grid_array)
    cv2.imshow("Grid", grid_visual(grid_array))

    print("Instructions:")
    print("- 왼쪽 클릭 또는 드래그: 0과 1 반전")
    print("- s 키: 저장")
    print("- q 키: 종료")

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_grid(grid_array)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
