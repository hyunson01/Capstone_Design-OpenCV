import sys
import os

# D:\git\MAPF-ICBS\code 경로를 추가
sys.path.append(r"D:\git\MAPF-ICBS\code")

import cv2
import numpy as np
from grid import load_grid
from visual import grid_visual, cell_size
<<<<<<< HEAD
from cbs_manager import CBSManager
=======
from cbs_manager import CBSManager  # 앞서 만든 CBSManager 사용
>>>>>>> 2a2ebcd35adb08fa2b43280641b59c6f47880fb5
from visualize import Animation

# 전역 변수
start_points = []
goal_points = []
paths = []
current_agent = 0

# 마우스 콜백 함수
def mouse_event(event, x, y, flags, param):
    global start_points, goal_points, paths, current_agent

    row, col = y // cell_size, x // cell_size
    if not (0 <= row < 12 and 0 <= col < 12):
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Start set at ({row}, {col})")
        start_points.append((row, col))
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Goal set at ({row}, {col})")
        goal_points.append((row, col))

    if len(start_points) == len(goal_points) and len(start_points) > len(paths):
        print("Calculating path...")
        grid_array = load_grid()
        map_array = grid_array.astype(bool)

<<<<<<< HEAD
        manager = CBSManager(solver_type="CBS", disjoint=True, visualize_result=False)
        manager.load_instance(map_array, start_points, goal_points)
        new_paths = manager.run()
        # new_paths = apply_start_delays(new_paths, starts, delays)
=======
        manager = CBSManager(solver_type="ICBS", disjoint=True, visualize_result=False)
        manager.load_instance(map_array, start_points, goal_points)
        new_paths = manager.run()
>>>>>>> 2a2ebcd35adb08fa2b43280641b59c6f47880fb5

        if new_paths:
            paths.clear()
            paths.extend(new_paths)
        else:
            print("No solution found.")


# 경로 시각화용 색상
COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
    (255, 128, 0), (255, 0, 0), (128, 0, 255), (255, 0, 255), (128, 128, 128)
]

def draw_paths(vis_img, paths):
    for idx, path in enumerate(paths):
        color = COLORS[idx % len(COLORS)]
        for pos in path:
            r, c = pos
            x, y = c * cell_size, r * cell_size
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (x, y), (x + cell_size, y + cell_size), color, -1)
            cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
<<<<<<< HEAD
            
            
def apply_start_delays(paths, starts, delays):
    delayed_paths = []
    for i, path in enumerate(paths):
        delay = delays[i]
        hold = [starts[i]] * delay
        delayed_paths.append(hold + path)
    return delayed_paths
=======
>>>>>>> 2a2ebcd35adb08fa2b43280641b59c6f47880fb5


def main():
    global start_points, goal_points, paths
    grid_array = load_grid()
    cv2.namedWindow("CBS Grid")
    cv2.setMouseCallback("CBS Grid", mouse_event)

    while True:
        vis = grid_visual(grid_array.copy())
        draw_paths(vis, paths)

        for i, pt in enumerate(start_points):
            x, y = pt[1] * cell_size, pt[0] * cell_size
            cv2.circle(vis, (x + cell_size//2, y + cell_size//2), 5, (0, 255, 0), -1)
            cv2.putText(vis, f"S{i}", (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        for i, pt in enumerate(goal_points):
            x, y = pt[1] * cell_size, pt[0] * cell_size
            cv2.circle(vis, (x + cell_size//2, y + cell_size//2), 5, (0, 0, 255), -1)
            cv2.putText(vis, f"G{i}", (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        cv2.imshow("CBS Grid", vis)
        key = cv2.waitKey(50)

        if key == ord('q'):  # Q 키로 종료
            break
        elif key == ord('r'):
            print("Reset all")
            start_points.clear()
            goal_points.clear()
            paths.clear()
        elif key == ord('a'):
            if paths:
                print("Playing animation of last CBS result...")
                animation = Animation(load_grid().astype(bool), start_points, goal_points, paths)
                animation.show()
<<<<<<< HEAD
                animation.save("demo.gif", speed=1.0)
=======
>>>>>>> 2a2ebcd35adb08fa2b43280641b59c6f47880fb5
            else:
                print("No paths available to animate.")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
