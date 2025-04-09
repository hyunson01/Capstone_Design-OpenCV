def generate_movement_commands(agent_paths, step_size=10):
    directions = {
        (0, 1): "right",
        (0, -1): "left",
        (1, 0): "down",
        (-1, 0): "up"
    }

    for agent, path in agent_paths.items():
        print(f"\nAgent {agent} movement commands:")
        
        prev_x, prev_y = path[0]
        prev_dir = "down"  # 모든 로봇이 처음 '아래쪽'을 보고 있다고 가정

        for i in range(1, len(path)):
            x, y = path[i]
            dx, dy = x - prev_x, y - prev_y  # 방향 벡터 계산

            if (dx, dy) not in directions:
                print(f"Warning: Unexpected move ({dx}, {dy}) at step {i} for agent {agent}")
                continue

            current_dir = directions[(dx, dy)]

            # 회전 명령이 필요한 경우
            if prev_dir and prev_dir != current_dir:
                if (prev_dir, current_dir) in [("up", "right"), ("right", "down"), ("down", "left"), ("left", "up")]:
                    print(f"{agent}: rotate: right")
                else:
                    print(f"{agent}: rotate: left")

            print(f"{agent}: move: {step_size}")
            
            prev_x, prev_y = x, y
            prev_dir = current_dir  # 현재 방향 업데이트
        
        print("end")  # 한 에이전트의 명령 종료
