# simulator.py
import cv2
import numpy as np
from fake_mqtt import FakeMQTTBroker

class Simulator:
    def __init__(self, map_array, colors, cell_size=50):
        self.map_array = map_array
        self.colors = colors
        self.cell_size = cell_size
        self.robots = {}
        self.vis = self.create_grid()
        self.paused = True
        self.robot_info = {}
        self.robot_past_paths = {}

    # 로봇 추가
    def add_robot(self, robot_id, broker, start_pos=(0, 0)):
        if robot_id in self.robots:
            return self.robots[robot_id]  # 이미 있으면 기존 객체 반환

        robot = Robot(robot_id, broker, start_pos)
        self.robots[robot_id] = robot
        print(f"Simulator: 로봇 {robot_id} 추가 완료. 시작 위치: {start_pos}")
        self.robot_info[robot_id] = {'path': None, 'goal': None, 'start': start_pos}
        return robot

    # 맵 그리기
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

    # 로봇 그리기
    def draw_robots(self, vis):
        for robot in self.robots.values():
            pos = robot.get_position()
            cx = int(pos[1] * self.cell_size + self.cell_size // 2)
            cy = int(pos[0] * self.cell_size + self.cell_size // 2)
            
            color = self.colors[robot.robot_id % len(self.colors)]
            cv2.circle(vis, (cx, cy), self.cell_size // 3, color, -1)
                   
    # # 로봇 출발지, 도착지 그리기
    # def draw_start_goal(self, vis):
    #     overlay = vis.copy()
    #     for robot_id, info in self.robot_info.items():
    #         start = info.get('start')
    #         goal = info.get('goal')
    #         color = self.colors[robot_id % len(self.colors)]
            
    #         # 🟪 출발지 그리기 (네모)
    #         if start:
    #             top_left = (start[1] * self.cell_size + self.cell_size // 4,
    #                         start[0] * self.cell_size + self.cell_size // 4)
    #             bottom_right = (start[1] * self.cell_size + self.cell_size * 3 // 4,
    #                             start[0] * self.cell_size + self.cell_size * 3 // 4)
    #             cv2.rectangle(overlay, top_left, bottom_right, color, -1)

    #         # 🔺 도착지 그리기 (삼각형)
    #         if goal:
    #             center_x = goal[1] * self.cell_size + self.cell_size // 2
    #             center_y = goal[0] * self.cell_size + self.cell_size // 2
    #             pts = np.array([
    #                 (center_x, center_y - self.cell_size // 4),
    #                 (center_x - self.cell_size // 4, center_y + self.cell_size // 4),
    #                 (center_x + self.cell_size // 4, center_y + self.cell_size // 4)
    #             ], np.int32)
    #             cv2.fillPoly(overlay, [pts], color)
                
    #     # ✅ 반투명으로 합치기
    #     cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
       
    # 로봇 경로 그리기
    # def draw_paths(self, vis):
    #     overlay = vis.copy()
    #     for robot_id, info in self.robot_info.items():
    #         color = self.colors[robot_id % len(self.colors)]

    #         past_path = self.robot_past_paths.get(robot_id, [])
    #         current_path = info['path'] if info['path'] else []

    #         # 🔥 경로 연결할 리스트
    #         full_path = []

    #         if past_path:
    #             full_path.extend(past_path)

    #         if current_path:
    #             # 🔥 지나온 마지막 위치와 새로운 경로 첫 위치가 다르면, 연결 끊기
    #             if not past_path or past_path[-1] == current_path[0]:
    #                 full_path.extend(current_path)
    #             else:
    #                 print(f"Robot {robot_id}: Path discontinuity detected. Not connecting past and current paths.")
    #                 # 지나온 경로 그린 다음, 새 경로는 따로 그린다.

    #         # 🔥 경로 그리기
    #         for i in range(1, len(full_path)):
    #             p1 = (full_path[i-1][1] * self.cell_size + self.cell_size // 2, full_path[i-1][0] * self.cell_size + self.cell_size // 2)
    #             p2 = (full_path[i][1] * self.cell_size + self.cell_size // 2, full_path[i][0] * self.cell_size + self.cell_size // 2)
    #             cv2.line(overlay, p1, p2, color, thickness=3)

    #     cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

    # 한 프레임 그리기
    def run_once(self):
        self.vis = self.create_grid()  # 배경(맵) 먼저 그림
        
        # self.draw_paths(self.vis)          # 경로 먼저 그리기
        # self.draw_start_goal(self.vis)      # 출발지, 도착지 그리기
        self.draw_robots(self.vis)                  # 로봇(보간 이동) 그리기
        
        if not self.paused:
            self.tick()  # 로봇 이동 처리 및 위치 기록
        
        cv2.imshow("Simulator", self.vis)
    
    # 로봇 경로 보간
    # def get_interpolated_position(self):
    #     if not self.path or self.current_index >= len(self.path) - 1:
    #         return self.path[-1]

    #     current_pos = np.array(self.path[self.current_index])
    #     next_pos = np.array(self.path[self.current_index + 1])
    #     progress = self.substep / self.substeps_per_move
    #     interp_pos = (1 - progress) * current_pos + progress * next_pos
    #     return interp_pos
    
    # 로봇 한 틱 이동
    def tick(self):
        for robot in self.robots.values():
            robot.tick()
            
            pos = tuple(map(int, robot.get_position()))
            if robot.robot_id not in self.robot_past_paths:
                self.robot_past_paths[robot.robot_id] = []
            if not self.robot_past_paths[robot.robot_id] or self.robot_past_paths[robot.robot_id][-1] != pos:
                self.robot_past_paths[robot.robot_id].append(pos)
                    
    def get_robot_current_positions(self):
        positions = {}
        for robot in self.robots.values():
            positions[robot.robot_id] = robot.get_position()
        return positions


class Robot:
    def __init__(self, robot_id, broker, start_pos):
        self.robot_id = robot_id
        self.broker = broker
        self.position = start_pos  # (row, col)
        self.moving = False         # 현재 1칸 이동 중인지
        self.start_pos = start_pos  # 보간 시작 좌표
        self.target_pos = start_pos # 보간 목표 좌표
        self.progress = 0.0         # 0.0~1.0 보간 진행도
        self.speed = 0.1            # 1 tick당 이동 비율 (ex. 0.1 → 10 tick 동안 1칸 이동)
        self.direction = "north"   # 초기 방향
        self.current_command = None
        self.command_queue = []
        self.broker.subscribe(f"robot/{self.robot_id}/move", self.on_receive_command)
        
        print(f"Robot {self.robot_id}: 구독 시작 (토픽: robot/{self.robot_id}/move)")
    
    def set_path(self, path):
        self.path = path
        self.current_index = 0
        self.progress = 0.0

    def on_receive_command(self, compressed_command):
        parsed = self.parse_compressed_command(compressed_command)
        if self.moving or self.current_command is not None:
            print(f"[Robot {self.robot_id}] 이동 중 → 기존 명령 유지, queue 덮어쓰기")
            self.command_queue = parsed  # 기존 대기열 교체 (덮어쓰기)
        else:
            # 아무것도 진행 중이 아니면 바로 시작
            print(f"[Robot {self.robot_id}] 정지 상태 → 명령 즉시 실행")
            self.current_command = parsed.pop(0) if parsed else None
            self.command_queue = parsed

    def execute_command(self, command):
        if command == "forward":
            self.move_forward()
        elif command == "left":
            self.turn_left()
        elif command == "right":
            self.turn_right()
        elif command == "stop":
            print(f"[Robot {self.robot_id}] 정지.")
        else:
            print(f"[Robot {self.robot_id}] 알 수 없는 명령: {command}")
            
    def parse_compressed_command(self, compressed_command):
        result = []
        i = 0
        while i < len(compressed_command):
            cmd = compressed_command[i]
            i += 1
            count = 0
            # 숫자가 있을 경우 숫자 읽기
            while i < len(compressed_command) and compressed_command[i].isdigit():
                count = count * 10 + int(compressed_command[i])
                i += 1
            count = count if count > 0 else 1  # 없으면 1회
            result.extend([cmd] * count)
        return result

    def move_forward(self):
        x, y = self.position
        if self.direction == "north":
            next_pos = (x - 1, y)
        elif self.direction == "east":
            next_pos = (x, y + 1)
        elif self.direction == "south":
            next_pos = (x + 1, y)
        elif self.direction == "west":
            next_pos = (x, y - 1)
        
        self.start_pos = self.position
        self.target_pos = next_pos
        self.progress = 0.0
        self.moving = True
        # print(f"[Robot {self.robot_id}] 앞으로 이동 준비: {self.start_pos} -> {self.target_pos}")


    def turn_left(self):
        directions = ["north", "west", "south", "east"]
        idx = directions.index(self.direction)
        self.direction = directions[(idx + 1) % 4]
        # print(f"[Robot {self.robot_id}] 왼쪽 회전 → 방향: {self.direction}")

    def turn_right(self):
        directions = ["north", "east", "south", "west"]
        idx = directions.index(self.direction)
        self.direction = directions[(idx + 1) % 4]
        # print(f"[Robot {self.robot_id}] 오른쪽 회전 → 방향: {self.direction}")
        
    def tick(self):
        if self.moving:
            self.progress += self.speed
            if self.progress >= 1.0:
                self.progress = 1.0
                self.position = self.target_pos
                self.moving = False
        else:
            if self.current_command is None and self.command_queue:
                self.current_command = self.command_queue.pop(0)

            if self.current_command:
                self.execute_compressed_command(self.current_command)
                self.current_command = None

    
    def get_position(self):
        if self.moving:
            current = np.array(self.start_pos)
            target = np.array(self.target_pos)
            return (1 - self.progress) * current + self.progress * target
        else:
            return self.position


    def execute_compressed_command(self, code):
        if code == 'f':
            self.move_forward()
        elif code == 'l':
            self.turn_left()
        elif code == 'r':
            self.turn_right()
        elif code == 's':
            print(f"[Robot {self.robot_id}] 정지.")
        else:
            print(f"[Robot {self.robot_id}] 알 수 없는 명령어: {code}")
            
