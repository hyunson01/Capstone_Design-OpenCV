import sys
import os
import random

# D:\git\MAPF-ICBS\code 경로를 추가
sys.path.append(r"D:\git\Capstone_Design-OpenCV\MAPF-ICBS\code")

import cv2
import numpy as np
from grid import load_grid
from visual import grid_visual, cell_size
from cbs.agent import Agent
from visualize import Animation
# from simulator import Simulator
# from fake_mqtt import FakeMQTTBroker
from path_to_commands import path_to_commands
from commandSendTest2 import CommandSet
from cbs.pathfinder import PathFinder
from config import COLORS
import json


# 전역 변수
agents = []
paths = []
current_agent = 0
manager = None
# sim = None
# broker = FakeMQTTBroker()
pathfinder = None
PRESET_IDS = [1,3, 4]

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

def create_agent(start=None, goal=None, delay=None, agent_id=None):
    if agent_id is None:
        agent_id = len(agents)
    if delay is None:
        delay = random.randint(0, 5)
    return Agent(id=agent_id, start=start, goal=goal, delay=delay)

def compute_cbs():
    global paths, pathfinder

    grid_array = load_grid()

    # PathFinder 초기화
    if pathfinder is None:
        pathfinder = PathFinder(grid_array)

    # 경로 계산
    new_agents = pathfinder.compute_paths(agents)
    new_paths = [agent.get_final_path() for agent in new_agents]

    if not new_paths:
        print("No solution found.")
        return

    paths.clear()
    paths.extend(new_paths)

    print("Paths updated via PathFinder.")

    try:
        command_sets = []
        for agent in new_agents:
            commands = path_to_commands(agent.get_final_path(), initial_dir="north")
            command_set = CommandSet(str(agent.id), commands)
            command_sets.append(command_set)
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
    mapping = {
        "forward": "f",
        "left": "l",
        "right": "r",
        "stop": "s"
    }
    
    if not commands:
        return ''
    
    result = []
    prev = mapping[commands[0]]
    count = 1
    
    for cmd in commands[1:]:
        code = mapping[cmd]
        if code == prev:
            count += 1
        else:
            if count > 1:
                result.append(f"{prev}{count}")
            else:
                result.append(prev)
            prev = code
            count = 1
    # 마지막 명령어 처리
    if count > 1:
        result.append(f"{prev}{count}")
    else:
        result.append(prev)
    
    return ''.join(result)

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

def direction_between(pos1, pos2):
    r1, c1 = pos1
    r2, c2 = pos2
    if r1 == r2 and c1 + 1 == c2:
        return "east"
    elif r1 == r2 and c1 - 1 == c2:
        return "west"
    elif c1 == c2 and r1 + 1 == r2:
        return "south"
    elif c1 == c2 and r1 - 1 == r2:
        return "north"
    else:
        raise ValueError(f"Invalid move from {pos1} to {pos2}")
    
def main():
    global agents, paths, manager
    grid_array = load_grid()
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
                animation = Animation(load_grid().astype(bool),
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
            compute_cbs()
            
    cv2.destroyAllWindows()

# def _random_goal(self, avoid: tuple[int, int]) -> tuple[int, int]:
#         options = [cell for cell in self.valid_cells if cell != avoid]
#         return random.choice(options) if options else avoid


if __name__ == '__main__':
    main()
