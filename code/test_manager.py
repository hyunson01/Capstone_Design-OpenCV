import sys
import os

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
from fake_mqtt import FakeMQTTClient

# 전역 변수
agents = []
paths = []
current_agent = 0
manager = None
sim = None
client = FakeMQTTClient()

# 마우스 콜백 함수
def mouse_event(event, x, y, flags, param):
    global agents, paths, manager, sim

    row, col = y // cell_size, x // cell_size
    
    if not (0 <= row < 12 and 0 <= col < 12):
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        agent_id = len(agents)
        print(f"Start set at ({row}, {col})")
        agent = Agent(id=agent_id, start=(row, col), goal=None, delay=0)
        agents.append(agent)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # ✅ Goal 클릭: 목표 추가
        if agents:
            print(f"Goal set at ({row}, {col})")
            for agent in agents:
                if agent.goal is None:
                    agent.goal = (row, col)
                    break
                
            if all(agent.start is not None and agent.goal is not None for agent in agents):
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

def draw_paths(vis_img, paths):
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


def compute_cbs(sim=None):
    """
    sim=None 일 때: 전역 paths를 갱신
    sim!=None 일 때: sim.pending_paths에 새 경로를 세팅하고 시뮬레이터를 재개
    """
    # CBS 계산 전 시뮬레이터 일시정지
    if sim:
        sim.paused = True

    # 현재 에이전트 위치(start/goal)로부터 CBS 재계산
    grid_array = load_grid()
    map_array = grid_array.astype(bool)
    manager = CBSManager(solver_type="CBS", disjoint=True, visualize_result=False)
    manager.load_instance(map_array, agents)
    new_paths = manager.run()

    if not new_paths:
        print("No solution found.")
    else:
        if sim:
            # 시뮬레이터용: pending_paths 설정 후 내부 시간 리셋
            sim.set_pending_paths(new_paths)
            sim.time_step = 0
            sim.substep = 0
            print("New CBS paths ready!")
            sim.paused = False
        else:
            # 마우스 이벤트용: 전역 paths 갱신
            paths.clear()
            paths.extend(new_paths)
            print("Paths updated via mouse_event.")

def main():
    global agents, paths, manager
    grid_array = load_grid()
    cv2.namedWindow("CBS Grid")
    cv2.setMouseCallback("CBS Grid", mouse_event)

    sim = Simulator(grid_array.astype(bool), agents, colors=COLORS)

    while True:
        vis = grid_visual(grid_array.copy())
        draw_paths(vis, paths)

        for agent in agents:
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
        
        #키보드 입력처리리
        key = cv2.waitKey(50)

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
    
    

if __name__ == '__main__':
    main()
