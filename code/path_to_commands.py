def direction_between(pos1, pos2):
    """두 좌표를 보고 방향을 계산."""
    dy = pos2[0] - pos1[0]
    dx = pos2[1] - pos1[1]
    if dy == -1 and dx == 0:
        return "north"
    elif dy == 1 and dx == 0:
        return "south"
    elif dy == 0 and dx == 1:
        return "east"
    elif dy == 0 and dx == -1:
        return "west"
    else:
        raise ValueError(f"Invalid move from {pos1} to {pos2}")

def compute_turns(current_dir, target_dir):
    """현재 방향과 목표 방향을 비교해 'left', 'right', 'forward'를 계산."""
    directions = ["north", "east", "south", "west"]
    idx_current = directions.index(current_dir)
    idx_target = directions.index(target_dir)

    diff = (idx_target - idx_current) % 4
    if diff == 0:
        return []
    elif diff == 1:
        return ["right"]
    elif diff == 3:
        return ["left"]
    elif diff == 2:
        return ["right", "right"]  # 180도 턴
    else:
        return []

def path_to_commands(path, initial_dir="north"):
    """
    경로를 받아서 이동 명령어 시퀀스로 변환.
    path: [(x1, y1), (x2, y2), (x3, y3), ...]
    initial_dir: 시작 방향
    """
    if len(path) < 2:
        return []

    commands = []
    curr_dir = initial_dir
    for i in range(1, len(path)):
        curr_pos = path[i-1]
        next_pos = path[i]
        target_dir = direction_between(curr_pos, next_pos)

        # 필요한 방향으로 회전
        turns = compute_turns(curr_dir, target_dir)
        commands.extend(turns)

        # 전진
        commands.append("forward")

        # 방향 업데이트
        curr_dir = target_dir

    commands.append("stop")  # 마지막에 정지
    return commands