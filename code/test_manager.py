import sys
import os
import random

# D:\git\MAPF-ICBS\code 경로를 추가
sys.path.append(r"D:\git\Capstone_Design-OpenCV\MAPF-ICBS\code")

import cv2
import numpy as np
from grid import load_grid
from visual import grid_visual, cell_size
from cbs.cbs_manager import CBSManager
from cbs.agent import Agent
from visualize import Animation
from simulator import Simulator
from fake_mqtt import FakeMQTTBroker
from path_to_commands import path_to_commands

# 전역 변수
agents = []
paths = []
current_agent = 0
manager = None
sim = None
broker = FakeMQTTBroker()

# 마우스 콜백 함수
def mouse_event(event, x, y, flags, param):
    global agents, paths, manager, sim
    
    row, col = y // cell_size, x // cell_size
    if not (0 <= row < 12 and 0 <= col < 12):
        return

    if event == cv2.EVENT_LBUTTONDOWN:  # 좌클릭 (출발지)
        print(f"Start set at ({row}, {col})")
        # 출발지가 없는 agent 찾기
        for agent in agents:
            if agent.start is None and agent.goal is not None:
                agent.start = (row, col)
                break
        else:
            # 출발지가 있는 agent 중 위치 이동
            for agent in agents:
                if agent.start is not None and agent.goal is None:
                    agent.start = (row, col)
                    break
            else:
                # 아예 없으면 새로 생성
                agent = create_agent(start=(row, col))
                agents.append(agent)


    elif event == cv2.EVENT_RBUTTONDOWN:  # 우클릭 (도착지)
        print(f"Goal set at ({row}, {col})")
        # 도착지가 없는 agent 찾기
        for agent in agents:
            if agent.goal is None and agent.start is not None:
                agent.goal = (row, col)
                break
        else:
            # 도착지가 있는 agent 중 위치 이동
            for agent in agents:
                if agent.goal is not None and agent.start is None:
                    agent.goal = (row, col)
                    break
            else:
                # 아예 없으면 새로 생성
                agent = create_agent(goal=(row, col))
                agents.append(agent)


    # ★ 출발지와 도착지가 모두 있는 agent가 하나라도 완성됐으면
    if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
        if agents and all(agent.start is not None and agent.goal is not None for agent in agents):
            compute_cbs(sim)


# 경로 시각화용 색상
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
    (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
    (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192)
]

def create_agent(start=None, goal=None, delay=None, agent_id=None):
    if agent_id is None:
        agent_id = len(agents)
    if delay is None:
        delay = random.randint(0, 5)
    return Agent(id=agent_id, start=start, goal=goal, delay=delay)


def compute_cbs(sim=None):
    global broker, manager, paths

    if sim:
        # sim.paused = True  # 일단 멈추고
        current_positions = sim.get_robot_current_positions()  # 🔥 현재 위치 가져오기

    grid_array = load_grid()
    map_array = grid_array.astype(bool)
    
    manager = CBSManager(solver_type="CBS", disjoint=True, visualize_result=False)

    new_agents = []

    # 기존 agents를 그대로 쓰는게 아니라 새로 start, goal을 정의
    for agent in agents:
        if sim and agent.id in current_positions:
            current_start = tuple(map(int, current_positions[agent.id]))  # 현재 위치 (반올림/정수 변환)
        else:
            current_start = agent.start  # 없으면 기존 start

        new_agent = Agent(id=agent.id, start=current_start, goal=agent.goal, delay=agent.delay)
        new_agents.append(new_agent)

    manager.load_instance(map_array, new_agents)

    new_paths = manager.run()

    if not new_paths:
        print("No solution found.")

    paths.clear()
    paths.extend(new_paths)

    if sim:
        past_paths_backup = sim.robot_past_paths.copy()  # 🔥 지나온 경로 백업
        sim.robots.clear()
        sim.robot_past_paths = past_paths_backup 

        print("New CBS paths ready! Sending commands to robots...")
        for agent in new_agents:
            robot = sim.add_robot(agent.id, broker, start_pos=agent.start)
            sim.robot_info[robot.robot_id]['path'] = agent.get_final_path()
            sim.robot_info[robot.robot_id]['goal'] = agent.goal
        
        for agent_id, path in enumerate(new_paths):
            commands = path_to_commands(path, initial_dir="north")
            compressed_cmd = compress_commands(commands)
            topic = f"robot/{agent_id}/move"
            broker.publish(topic, compressed_cmd)
        
        # sim.paused = False
    else:
        print("Paths updated via mouse_event.")

        
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
    if sim:
        for robot_id, past_path in sim.robot_past_paths.items():
            color = COLORS[robot_id % len(COLORS)]
            for pos in past_path:
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
    global agents, paths, manager, sim
    grid_array = load_grid()
    cv2.namedWindow("CBS Grid")
    cv2.setMouseCallback("CBS Grid", mouse_event)

    sim = Simulator(grid_array.astype(bool), colors=COLORS)

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
        
        sim.run_once()
        if not sim.paused:
            sim.tick()
        
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
        elif key == ord(' '):  # ✅ Spacebar 눌러서 일시정지
            sim.paused = not sim.paused
            print("Paused" if sim.paused else "Resumed")
        elif key == ord('c'):  # 'c' 키로 CBS 재계산
            compute_cbs(sim)
            
    cv2.destroyAllWindows()
    
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

    

if __name__ == '__main__':
    main()
