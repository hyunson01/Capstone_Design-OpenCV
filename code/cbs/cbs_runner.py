import random

from config import cbs_path, board_width_cm, board_height_cm, grid_width, grid_height, cell_size
from cbs.cbs_manager import CBSManager

def run_cbs_manager(grid_array, tag_info):
    """
    cbs_manager를 사용해 CBS를 직접 실행하고, 결과를 relay에 저장
    """
    # Step 1: grid_array와 tag_info로 starts, goals, ids를 만든다
    rows, cols = grid_array.shape

    start_points = []
    goal_points = []
    agent_ids = []

    valid_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid_array[r, c] == 0:
                valid_cells.append((r, c))

    for idx, (tag_id, data) in enumerate(tag_info.items()):
        x_cm, y_cm = data["coordinates"]

        tag_grid_x = int(x_cm * grid_width / board_width_cm)
        tag_grid_y = int(y_cm * grid_height / board_height_cm)
        start_col = tag_grid_x // cell_size
        start_row = tag_grid_y // cell_size

        start_row = max(0, min(rows - 1, start_row))
        start_col = max(0, min(cols - 1, start_col))

        if valid_cells:
            dest_row, dest_col = random.choice(valid_cells)
        else:
            dest_row, dest_col = start_row, start_col

        start_points.append((start_row, start_col))
        goal_points.append((dest_row, dest_col))
        agent_ids.append(tag_id)  # 여기서 tag_id를 agent id로 사용

    # Step 2: CBSManager를 이용해 CBS 실행
    map_array = grid_array.astype(bool)  # 0: 이동 가능, 1: 장애물
    manager = CBSManager(solver_type="ICBS", disjoint=True, visualize_result=False)
    manager.load_instance(map_array, start_points, goal_points, agent_ids)
    manager.run()  # -> 내부에서 자동으로 path_relay에 저장

    # Step 3: 결과 가져오기
    agents = manager.get_agents()

    print("\n=== Extracted Agents ===")
    for agent in agents:
        print(agent)

    return agents
