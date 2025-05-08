import sys
import os
import random

# MAPF-ICBS\code 경로를 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ICBS_PATH = os.path.join(CURRENT_DIR, '..', 'MAPF-ICBS', 'code')
sys.path.append(os.path.normpath(ICBS_PATH))

import cv2
import numpy as np
from grid import load_grid
from visual import grid_visual, cell_size
from cbs.agent import Agent
from visualize import Animation
# from simulator import Simulator
# from fake_mqtt import FakeMQTTBroker
from commandSendTest2 import CommandSet
from cbs.pathfinder import PathFinder
from config import COLORS, grid_row, grid_col
import json

# 전역 변수
agents = []
paths = []
current_agent = 0
manager = None
# sim = None
# broker = FakeMQTTBroker()
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

#에이전트 초기화
def create_agent(start=None, goal=None, delay=None, agent_id=None):
    if agent_id is None:
        agent_id = len(agents)
    if delay is None:
        delay = random.randint(0, 5)
    return Agent(id=agent_id, start=start, goal=goal, delay=delay)

#CBS 계산
def compute_cbs():
    global paths, pathfinder, grid_array

    grid_array = load_grid(grid_row, grid_col)

    if pathfinder is None:
        pathfinder = PathFinder(grid_array)

    new_agents = pathfinder.compute_paths(agents)
    new_paths = [agent.get_final_path() for agent in new_agents]

    if not new_paths:
        print("No solution found.")
        return

    paths.clear()
    paths.extend(new_paths)

    print("Paths updated via PathFinder.")

    # 로봇 명령 전송
    command_sets = [CommandSet(str(agent.id), agent.get_final_path()) for agent in new_agents]

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


    # if sim:
    #     for agent in new_agents:
    #         robot = sim.add_robot(agent.id, broker, start_pos=agent.start)
    #         sim.robot_info[robot.robot_id]['path'] = agent.get_final_path()
    #         sim.robot_info[robot.robot_id]['goal'] = agent.goal
    #     sim.paused = False
    

def compress_commands(commands):
#     mapping = {
#         "forward": "f",
#         "left": "l",
#         "right": "r",
#         "stop": "s"
#     }
    
#     if not commands:
#         return ''
    
#     result = []
#     prev = mapping[commands[0]]
#     count = 1
    
#     for cmd in commands[1:]:
#         code = mapping[cmd]
#         if code == prev:
#             count += 1
#         else:
#             if count > 1:
#                 result.append(f"{prev}{count}")
#             else:
#                 result.append(prev)
#             prev = code
#             count = 1
#     # 마지막 명령어 처리
#     if count > 1:
#         result.append(f"{prev}{count}")
#     else:
#         result.append(prev)
    
#     return ''.join(result)
        return 

#경로 색칠용 코드
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
    
    # 2. 추가: sim.robot_past_paths에 저장된 지나간 경로도 색칠
    # if sim:
    #     for robot_id, past_path in sim.robot_past_paths.items():
    #         color = COLORS[robot_id % len(COLORS)]
    #         for pos in past_path:
    #             r, c = pos
    #             x, y = c * cell_size, r * cell_size
    #             overlay = vis_img.copy()
    #             cv2.rectangle(overlay, (x, y), (x + cell_size, y + cell_size), color, -1)
    #             cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
         
def apply_start_delays(paths, starts, delays):
    delayed_paths = []
    for i, path in enumerate(paths):
        delay = delays[i]
        hold = [starts[i]] * delay
        delayed_paths.append(hold + path)
    return delayed_paths
    
def main():
    global agents, paths, manager, grid_array
    grid_array = load_grid(grid_row, grid_col)
    cv2.namedWindow("CBS Grid")
    cv2.setMouseCallback("CBS Grid", mouse_event)

    # sim = Simulator(grid_array.astype(bool), colors=COLORS)

    while True:
        vis = grid_visual(grid_array.copy())
        draw_paths(vis, paths)

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
        
        # sim.run_once()
        # if not sim.paused:
        #     sim.tick()
        
        # 키보드 입력 처리
        key = cv2.waitKey(100)
        if key == ord('q'):
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
        # elif key == ord(' '):  # ✅ Spacebar 눌러서 일시정지
        #     sim.paused = not sim.paused
        #     print("Paused" if sim.paused else "Resumed")
        
        elif key == ord('c'):  # 'c' 키로 CBS 재계산
            if all(a.start and a.goal for a in agents):
                compute_cbs()
            else:
                print("⚠️  start 또는 goal이 비어 있는 에이전트가 있습니다.")
            
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
