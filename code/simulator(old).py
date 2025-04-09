import cv2
import numpy as np

class Simulator:
    def __init__(self, map_array, agents,colors, cell_size=50):
        self.map_array = map_array
        self.agents = agents
        self.colors = colors
        self.cell_size = cell_size
        self.time_step = 0
        self.substep = 0
        self.substeps_per_move = 20
        self.paused = True
        self.pending_new_paths = None

        self.vis = self.create_grid()
        
    def step(self):
        if not self.paused:
            self.substep += 1
            if self.substep >= self.substeps_per_move:
                self.time_step += 1
                self.substep = 0

                # ✅ 칸 이동이 끝났으면 경로 교체 체크
                if self.pending_new_paths is not None:
                    for agent, path in zip(self.agents, self.pending_new_paths):
                        agent.set_path(path)
                    self.pending_new_paths = None
                    self.time_step = 0
                    self.substep = 0
                    print("New CBS paths applied!")
            
    def run_once(self):
        self.vis = self.create_grid()
        self.draw_agents()
        self.step()
        cv2.imshow("Simulator", self.vis)

    def create_grid(self):
        rows, cols = self.map_array.shape
        vis = np.ones((rows * self.cell_size, cols * self.cell_size, 3), dtype=np.uint8) * 255
        for r in range(rows):
            for c in range(cols):
                if self.map_array[r, c]:
                    cv2.rectangle(vis,
                                  (c * self.cell_size, r * self.cell_size),
                                  ((c+1) * self.cell_size, (r+1) * self.cell_size),
                                  (0, 0, 0), -1)
        return vis

    def draw_agents(self):
        for agent in self.agents:
            path = agent.get_final_path()
            if not path:  # 아직 경로 없으면 무시
                continue

            # 현재 칸과 다음 칸 사이를 보간
            if self.time_step < len(path):
                current_pos = path[self.time_step]
            else:
                current_pos = path[-1]

            if self.time_step + 1 < len(path):
                next_pos = path[self.time_step + 1]
            else:
                next_pos = path[-1]

            # 🔥 보간 (Interpolation)
            progress = self.substep / self.substeps_per_move  # 0 ~ 1 사이 값
            interp_pos = (
                (1 - progress) * np.array(current_pos) + 
                progress * np.array(next_pos)
            )

            # 화면상 위치 변환
            cx = int(interp_pos[1] * self.cell_size + self.cell_size // 2)
            cy = int(interp_pos[0] * self.cell_size + self.cell_size // 2)

            # 원 그리기
            cv2.circle(self.vis, (cx, cy), self.cell_size // 3, (0, 0, 255), -1)
            
    def draw_start_goal(self):
        overlay = self.vis.copy()
        for agent in self.agents:
            color = self.colors[agent.id % len(self.colors)]

            start = agent.start
            goal = agent.goal

            if start:
                # 반투명 네모 (출발지)
                top_left = (start[1] * self.cell_size + self.cell_size // 4, start[0] * self.cell_size + self.cell_size // 4)
                bottom_right = (top_left[0] + self.cell_size // 2, top_left[1] + self.cell_size // 2)
                cv2.rectangle(overlay, top_left, bottom_right, color, -1)

            if goal:
                # 반투명 세모 (도착지)
                center = (goal[1] * self.cell_size + self.cell_size // 2, goal[0] * self.cell_size + self.cell_size // 2)
                pts = np.array([
                    (center[0], center[1] - self.cell_size // 4),
                    (center[0] - self.cell_size // 4, center[1] + self.cell_size // 4),
                    (center[0] + self.cell_size // 4, center[1] + self.cell_size // 4)
                ], np.int32)
                cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, 0.4, self.vis, 0.6, 0, self.vis)
        
        
    def draw_paths(self):
        overlay = self.vis.copy()
        for agent in self.agents:
            path = agent.get_final_path()
            color = self.colors[agent.id % len(self.colors)]
            for i in range(1, len(path)):
                p1 = (path[i-1][1] * self.cell_size + self.cell_size // 2, path[i-1][0] * self.cell_size + self.cell_size // 2)
                p2 = (path[i][1] * self.cell_size + self.cell_size // 2, path[i][0] * self.cell_size + self.cell_size // 2)
                cv2.line(overlay, p1, p2, color, thickness=4)
        cv2.addWeighted(overlay, 0.3, self.vis, 0.7, 0, self.vis)


    def draw_agents(self):
        self.draw_paths()
        self.draw_start_goal()
        for agent in self.agents:
            path = agent.get_final_path()
            if not path:
                continue

            if self.time_step < len(path):
                current_pos = path[self.time_step]
            else:
                current_pos = path[-1]

            if self.time_step + 1 < len(path):
                next_pos = path[self.time_step + 1]
            else:
                next_pos = path[-1]

            progress = self.substep / self.substeps_per_move
            interp_pos = (
                (1 - progress) * np.array(current_pos) + 
                progress * np.array(next_pos)
            )

            color = self.colors[agent.id % len(self.colors)]

            cx = int(interp_pos[1] * self.cell_size + self.cell_size // 2)
            cy = int(interp_pos[0] * self.cell_size + self.cell_size // 2)
            cv2.circle(self.vis, (cx, cy), self.cell_size // 3, color, -1)

    def get_agent_current_cell(self, agent):
        path = agent.get_final_path()
        if not path:
            return agent.start

        if self.time_step < len(path):
            return path[self.time_step]
        else:
            return path[-1]
        
    def set_pending_paths(self, new_paths):
        self.pending_new_paths = new_paths
        
    def run(self):
        while True:
            self.step()
            cv2.imshow("Simulator", self.vis)
            key = cv2.waitKey(100)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        
