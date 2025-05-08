import sys
import os

import cv2
import numpy as np
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ICBS_PATH = os.path.join(CURRENT_DIR, '..', 'MAPF-ICBS', 'code')
sys.path.append(os.path.normpath(ICBS_PATH))

from vision.camera import camera_open, frame_process
from vision.board import board_detect, perspective_transform, board_pts, board_origin, board_draw
from vision.apriltag import AprilTagDetector, cm_per_px
from vision.tracking import TrackingManager
from grid import load_grid
from visual import grid_visual, grid_tag_visual, info_tag, slider_create, cell_size
from config import tag_info, object_points, camera_matrix, dist_coeffs, COLORS, grid_row, grid_col
from movement.movement_generator import generate_movement_commands
from cbs.pathfinder import PathFinder
from commandSendTest2 import CommandSet
from cbs.agent import Agent
from vision.apriltag import transform_coordinates 
from visualize import Animation

# 전역 변수
agents = []
paths = []
current_agent = 0
manager = None
pathfinder = None
grid_array = None

# 사용할 ID 목록
PRESET_IDS = [1,2,3,4,5,6,7,8,9,10,11,12]  # 예시: 1~12까지의 ID 사용

# 마우스 콜백 함수
def mouse_event(event, x, y, flags, param):
    """
    좌클릭  : 출발지(start) 지정
    우클릭  : 도착지(goal)  지정
    - PRESET_IDS(예: [2, 4]) 두 개가 모두 완성되면 CBS 실행
    """
    global agents, paths, pathfinder

    row, col = y // cell_size, x // cell_size
    if not (0 <= row < 12 and 0 <= col < 12):
        return

    updated = False                 # ← 변경 여부 플래그
    complete_agents = [a for a in agents if a.start and a.goal]

    # ---------- 1. 출발지 클릭 ----------
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Start set at ({row}, {col})")

        # 1‑A. 이미 완성된 agent가 한도(PRESET_IDS)만큼이면 생성 제한
        if len(complete_agents) >= len(PRESET_IDS):
            print("더 이상 agent를 생성할 수 없습니다.")
            return
        
        if event == cv2.EVENT_LBUTTONDOWN and any(a.start == (row, col) for a in agents):
            return

        # 1‑B. goal‑only agent에 start 채우기
        for agent in agents:
            if agent.start is None and agent.goal is not None:
                agent.start = (row, col)
                updated = True
                break

        # 1‑C. start‑only agent의 start 덮어쓰기
        if not updated:
            for agent in agents:
                if agent.start is not None and agent.goal is None:
                    agent.start = (row, col)
                    updated = True
                    break

        # 1‑D. 둘 다 없으면 새 agent 생성
        if not updated:
            # 사용하지 않은 ID 선택
            used_ids = {a.id for a in agents}
            avail_ids = [pid for pid in PRESET_IDS if pid not in used_ids]
            if not avail_ids:
                print("더 이상 agent를 생성할 수 없습니다.")
                return
            new_id = avail_ids[0]
            agent = Agent(id=new_id, start=(row, col), goal=None, delay=0)
            agents.append(agent)
            updated = True

    # ---------- 2. 도착지 클릭 ----------
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Goal set at ({row}, {col})")

        # 2‑A. 이미 완성된 agent가 한도만큼이면 생성 제한
        if len(complete_agents) >= len(PRESET_IDS):
            print("더 이상 agent를 생성할 수 없습니다.")
            return

        # 2‑B. start‑only agent에 goal 채우기
        for agent in agents:
            if agent.goal is None and agent.start is not None:
                agent.goal = (row, col)
                updated = True
                break

        # 2‑C. goal‑only agent의 goal 덮어쓰기
        if not updated:
            for agent in agents:
                if agent.goal is not None and agent.start is None:
                    agent.goal = (row, col)
                    updated = True
                    break

        # 2‑D. 둘 다 없으면 새 agent 생성 (goal‑only)
        if not updated:
            used_ids = {a.id for a in agents}
            avail_ids = [pid for pid in PRESET_IDS if pid not in used_ids]
            if not avail_ids:
                print("더 이상 agent를 생성할 수 없습니다.")
                return
            new_id = avail_ids[0]
            agent = Agent(id=new_id, start=None, goal=(row, col), delay=0)
            agents.append(agent)
            updated = True

    # ---------- 3. 공통 후처리 ----------
    if updated:
        target_ids = set(PRESET_IDS)  # ← PRESET_IDS 기반으로 변경
        ready_ids  = {a.id for a in agents if a.start and a.goal and a.id in target_ids}

        if ready_ids == target_ids:
            print(f"Agent {sorted(ready_ids)} 준비 완료. CBS 실행.")
            compute_cbs()

def update_agents_from_tags(tag_info):
    """
    Apriltag 정보(tag_info) → agents 리스트 반영.
    ① PRESET_IDS에 없는 태그는 무시
    ② 좌표가 ‘바뀐’ 경우에만 start 갱신 → 불필요한 CBS 재계산 방지
    """
    grid_tags = transform_coordinates(tag_info)          # cm → 셀 좌표
    for tag_id, data in grid_tags.items():
        if tag_id not in PRESET_IDS:                      # ①
            continue

        start_cell = data["grid_position"]                # (row, col)

        existing = next((a for a in agents if a.id == tag_id), None)
        if existing:                                      # 이미 agent 존재
            # ② 위치가 그대로면 아무것도 하지 않고 다음 tag로
            if existing.start == start_cell:
                continue
            existing.start = start_cell                   # 새 좌표로 갱신
        else:                                             # 처음 보는 tag
            agents.append(
                Agent(id=tag_id, start=start_cell, goal=None, delay=0)
            )


#CBS 계산
def compute_cbs():
    global paths, pathfinder, grid_array

    grid_array = load_grid(grid_row, grid_col)

    if pathfinder is None:
        pathfinder = PathFinder(grid_array)

    ready_agents = [a for a in agents if a.start and a.goal]
    if not ready_agents:
        print("⚠️  start·goal이 모두 지정된 에이전트가 없습니다.")
        return

    # ✅ 2) pathfinder에 ready_agents만 전달
    if pathfinder is None:
        pathfinder = PathFinder(load_grid())

    solved_agents = pathfinder.compute_paths(ready_agents)
    new_paths = [agent.get_final_path() for agent in solved_agents]

    if not new_paths:
        print("No solution found.")
        return

    paths.clear()
    paths.extend(new_paths)

    print("Paths updated via PathFinder.")

    # 로봇 명령 전송
    command_sets = [CommandSet(str(agent.id), agent.get_final_path()) for agent in solved_agents]

# 전송할 JSON 문자열을 미리 출력
    try:
        payload = json.dumps({"commands": [cs.to_dict() for cs in command_sets]}, indent=2, ensure_ascii=False)
        print("!!!전송 예정 명령 세트:")
        print(payload)
    except Exception as e:
        print(f"명령 세트 변환 중 오류 발생: {e}")

    # 실제 전송 시도
    try:
        CommandSet.send_command_sets(command_sets)
    except Exception as e:
        print(f"명령 전송 중 오류 발생: {e}")

def draw_paths(vis_img, paths):
    # 1. paths (CBS 경로) 색칠
    for idx, path in enumerate(paths):
        color = COLORS[idx % len(COLORS)]
        for pos in path:
            r, c = pos
            x, y = c * cell_size, r * cell_size
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (x, y), (x + cell_size, y + cell_size), color, -1)
            cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)

def apply_start_delays(paths, starts, delays):
    delayed_paths = []
    for i, path in enumerate(paths):
        delay = delays[i]
        hold = [starts[i]] * delay
        delayed_paths.append(hold + path)
    return delayed_paths

def main():
    global agents, paths, manager
    cap, fps = camera_open()
    frame_count = 0
    
    base_grid = load_grid(grid_row, grid_col)
    grid_array = base_grid.copy()

    slider_create()
    
    tracking_manager = TrackingManager(window_size=5)
    tag_detector = AprilTagDetector()
    
    cv2.namedWindow("CBS Grid")
    cv2.setMouseCallback("CBS Grid", mouse_event)

    while True:
        frame_count += 1
        time = frame_count / fps
        frame, gray = frame_process(cap, camera_matrix, dist_coeffs)
        vis = grid_visual(grid_array.copy())
        draw_paths(vis, paths)

        if frame is None:
            continue

        largest_rect = board_detect(gray)

        if largest_rect is not None:
            board_draw(frame, largest_rect)
            rect, board_width_px, board_height_px = board_pts(largest_rect)
            warped, warped_board_width_px, warped_board_height_px, warped_resized = perspective_transform(frame, rect, board_width_px, board_height_px)
            board_origin_tvec = board_origin(frame, rect[0])

            cm_per_pixel = cm_per_px(warped_board_width_px, warped_board_height_px)
            
            tags = tag_detector.tag_detect(gray)
            tag_detector.tags_process(tags, object_points, frame_count, board_origin_tvec, cm_per_pixel, frame, camera_matrix, dist_coeffs)
            tracking_manager.update_all(tag_info, time)
            
            update_agents_from_tags(tag_info) 

            info_tag(frame, tag_info)
            
            cv2.imshow("Warped Perspective", warped_resized)

        for agent in agents:
            if agent.start:
                x, y = agent.start[1] * cell_size, agent.start[0] * cell_size
                cv2.circle(vis, (x + cell_size//2, y + cell_size//2), 5, (0, 255, 0), -1)
                cv2.putText(vis, f"S{agent.id}", (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        for agent in agents:
            if agent.goal:
                x, y = agent.goal[1] * cell_size, agent.goal[0] * cell_size
                cv2.circle(vis, (x + cell_size//2, y + cell_size//2), 5, (0, 0, 255), -1)
                cv2.putText(vis, f"G{agent.id}", (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        cv2.imshow("CBS Grid", vis)
        cv2.imshow("Detected Rectangle", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):  # 'q' 키 -> 종료 (저장 없이)
            break
        elif key == ord('r'):
            print("Reset all")
            agents.clear()
            paths.clear()
        elif key == ord('a'):
            if paths:
                print("Playing animation of last CBS result...")
                animation = Animation(grid_array.astype(bool),
                              [agent.start for agent in agents],
                              [agent.goal for agent in agents],
                              [agent.get_final_path() for agent in agents])
                animation.show()
                animation.save("demo.gif", speed=1.0)
            else:
                print("No paths available to animate.")
        elif key == ord('m'):
            if manager:
                print("--- Current Agents ---")
                print(manager.get_agents())  # 그대로 OK
            else:
                print("No CBSManager initialized yet.")
        elif key == ord('c'):  # 'c' 키로 CBS 재계산
            if all(a.start and a.goal for a in agents):
                compute_cbs()
            else:
                print("⚠️  start 또는 goal이 비어 있는 에이전트가 있습니다.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
