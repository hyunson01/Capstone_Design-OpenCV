# mission.py

import json
import numpy as np

from grid import load_grid
from cbs.pathfinder import PathFinder
from recieve_message import start_sequence

def compute_cbs():
    import threading
    global paths, pathfinder, grid_array

    # 1) 그리드 로드 및 PathFinder 초기화
    grid_array = load_grid(grid_row, grid_col)
    if pathfinder is None:
        pathfinder = PathFinder(grid_array)

    # 2) 준비된 에이전트 추출
    ready_agents = [a for a in agents if a.start and a.goal]
    if not ready_agents:
        print("⚠️  start·goal이 모두 지정된 에이전트를 찾을 수 없습니다.")
        return

    # 3) CBS 경로 계산
    solved_agents = pathfinder.compute_paths(ready_agents)
    new_paths = [agent.get_final_path() for agent in solved_agents]
    if not new_paths:
        print("No solution found.")
        return

    # 4) 전역 paths 갱신
    paths.clear()
    paths.extend(new_paths)
    print("Paths updated via PathFinder.")

    # 5) CBS 결과를 우리 로직으로 변환하여 직접 JSON 페이로드 생성 (즉시 publish는 하지 않음)
    payload_commands = []
    for agent in solved_agents:
        raw_path = agent.get_final_path()
        init_hd = {'north':0, 'east':1, 'south':2, 'west':3}.get(
            getattr(agent, 'direction', 'north'),
            0
        )
        cmds      = path_to_commands(raw_path, init_hd)
        payload_commands.append({
            "robot_id":      str(agent.id),
            "command_count": len(cmds),
            "command_set":   [{ "command": c['command'] } for c in cmds]
        })

    # **여기서 디버그**  
    print("DEBUG: payload_commands =", payload_commands)

    # 6-A) 회전 오차 보정
    for cmd_info in payload_commands:
        rid_str = cmd_info["robot_id"]
        cmds = cmd_info["command_set"]

        try:
            rid = int(rid_str)
        except ValueError:
            continue

        yaw_error = 0.0
        if rid in tag_info:
            yaw_error = tag_info[rid].get("relative_angle_deg", 0.0)

        for cmd in cmds:
            raw = cmd["command"]
            if raw.startswith("R") or raw.startswith("L"):
                angle_part = raw[1:]
                suffix = ""
                if "_" in angle_part:
                    angle_str, suffix = angle_part.split("_", 1)
                else:
                    angle_str = angle_part

                try:
                    base_angle = float(angle_str)
                except ValueError:
                    continue

                if raw.startswith("R"):
                    corrected = base_angle - yaw_error
                    corrected = max(0, round(corrected, 1))
                    cmd["command"] = f"R{corrected}" + (f"_{suffix}" if suffix else "")
                elif raw.startswith("L"):
                    corrected = base_angle + yaw_error
                    corrected = max(0, round(corrected, 1))
                    cmd["command"] = f"L{corrected}" + (f"_{suffix}" if suffix else "")

                    
    # 6) payload_commands를 cmd_map으로 변환
    cmd_map = {
        cmd_info["robot_id"]: [c["command"] for c in cmd_info["command_set"]]
        for cmd_info in payload_commands
    }


    # 7) 순차 전송 시작
    print("▶ 순차 전송 시작:", cmd_map)
    start_sequence(cmd_map)


        
def send_emergency_stop():
    """모든 로봇(1~4번)에게 즉시 정지 명령 'S' 전송"""
    print("!! Emergency Stop 명령 전송: 'S' to robots 1~4")
    for rid in range(1, 5):               # 1,2,3,4
        topic = f"robot/{rid}/cmd"        # 각 로봇의 명령 토픽
        client.publish(topic, "S")
        print(f"  → Published to {topic}")
