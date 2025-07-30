# simulator.py
import cv2
import numpy as np
import time
import math
from fake_mqtt import FakeMQTTBroker
import threading
from datetime import datetime

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
        self.random_mode_enabled = False
        self.arrival_callback = None

    # 로봇 추가
    def add_robot(self, robot_id, broker, start_pos=(0, 0), direction="north"):
        if robot_id in self.robots:
            return self.robots[robot_id]  # 이미 있으면 기존 객체 반환

        robot = Robot(robot_id, broker, start_pos, direction=direction)
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
            
            # 삼각형 추가 (정면 방향 표시)
            dir_vecs = {
                "north": (0, -1),
                "east":  (1, 0),
                "south": (0, 1),
                "west":  (-1, 0)
            }
            dx, dy = robot.get_direction()
            length = self.cell_size // 4
            tip = (int(cx + dx * length), int(cy + dy * length))
            left = (int(cx + dy * length // 2), int(cy - dx * length // 2))
            right = (int(cx - dy * length // 2), int(cy + dx * length // 2))
            triangle_cnt = np.array([tip, left, right], np.int32)
            cv2.fillPoly(vis, [triangle_cnt], (0, 0, 0))  # 검은색 삼각형

    # 한 프레임 그리기
    def draw_frame(self):
        self.vis = self.create_grid()  # 배경(맵) 먼저 그림
        self.draw_robots(self.vis)                  # 로봇(보간 이동) 그리기
        cv2.imshow("Simulator", self.vis)

    def run_simulator(self, tick_interval=0.1):
        self.tick_interval = tick_interval
        def loop():
            self.running = True
            last_tick = time.time()

            while self.running:
                now = time.time()
                if now - last_tick >= tick_interval:
                    self.tick()
                    last_tick = now
                time.sleep(0.001)  # CPU 사용량 방지용 짧은 대기
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

    # 도착시 콜백 등록
    def register_arrival_callback(self, func):
        self.arrival_callback = func

    # 로봇 한 틱 이동
    def tick(self):
        for robot in self.robots.values():
            robot.tick_interval = self.tick_interval
            robot.tick()
            
            pos = tuple(map(int, robot.get_position()))
            if robot.robot_id not in self.robot_past_paths:
                self.robot_past_paths[robot.robot_id] = []
            if not self.robot_past_paths[robot.robot_id] or self.robot_past_paths[robot.robot_id][-1] != pos:
                self.robot_past_paths[robot.robot_id].append(pos)

            if getattr(self, "random_mode_enabled", False):
                info = self.robot_info[robot.robot_id]
                goal = info.get("goal")
                path = info.get("path")

                if path and len(path) > 0 and pos == path[-1]:
                    if self.arrival_callback:
                        self.arrival_callback(robot.robot_id, pos)
                    
    def get_robot_current_positions(self):
        positions = {}
        for robot in self.robots.values():
            positions[robot.robot_id] = robot.get_position()
        return positions

MOTION_DURATIONS = {
        "Move": 1.0,     # 전진
        "Stop": 1.0,      # 대기
        "Rotate": 1.0,     # 회전
    }

class Robot:
    def __init__(self, robot_id, broker, start_pos, direction="north"):
        self.robot_id = robot_id
        self.broker = broker
        self.next_move_extension = 0.0
        self.position = start_pos  # (row, col)
        self.current_deadline = None  # 현재 명령의 deadline
        
        # 이동 관련
        self.moving = False         # 현재 1칸 이동 중인지
            # 이동 보간
        self.start_pos = start_pos  # 보간 시작 좌표
        self.target_pos = start_pos # 보간 목표 좌표
        self.move_progress = 0.0         # 0.0~1.0 보간 진행도
        self.move_duration = MOTION_DURATIONS["Move"]           # 이동하는데 소요되는 초
    
            # 회전 관련
        self.direction = direction  # 초기 방향
            # 회전 보간
        self.rotating = False       # 회전 중인지 여부
        self.rotation_progress = 0.0
        self.rotation_duration = MOTION_DURATIONS["Rotate"]   # 회전하는데 소요되는 초
        self.rotation_dir = None      # "left" or "right"

        # 정지 관련
        self.stopping = False
        self.stop_progress = 0.0
        self.stop_duration = MOTION_DURATIONS["Stop"]  # 정지 시간 (초)

        self.current_command = None
        self.command_queue = []
        self.broker.subscribe(f"robot/{self.robot_id}/move", self.on_receive_command)
        
        print(f"Robot {self.robot_id}: 구독 시작 (토픽: robot/{self.robot_id}/move)")
    
    def set_path(self, path):
        self.path = path
        self.current_index = 0
        self.move_progress = 0.0

    def on_receive_command(self, command_list):
            parsed = []

            for raw in command_list:
                # 문자열 또는 dict 둘 다 처리
                s = raw.get("command", raw) if isinstance(raw, dict) else raw
                parts = s.split("@")
                cmd = parts[0]
                deadline = None

                if len(parts) > 1 and parts[1]:
                    # HHMMSSmmm → 오늘 날짜의 timestamp
                    dt = datetime.strptime(parts[1], "%H%M%S%f")
                    today = datetime.now()
                    dt = dt.replace(year=today.year,
                                    month=today.month,
                                    day=today.day)
                    deadline = dt.timestamp()

                parsed.append((cmd, deadline))

                if self.moving or self.rotating or self.current_command is not None:
                    self.command_queue = parsed
                else:
                    self.current_command = parsed.pop(0) if parsed else (None, None)
                    self.command_queue = parsed
            
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
        self.move_progress = 0.0
        self.moving = True
        # print(f"[Robot {self.robot_id}] 앞으로 이동 준비: {self.start_pos} -> {self.target_pos}")


    def turn(self, direction):  # direction: 'left' or 'right'
        if not self.rotating:
            self.rotating = True
            self.rotation_progress = 0.0
            self.rotation_dir = direction

        
    def tick(self):
        if self.stopping:
            self.stop_progress += self.tick_interval
            if self.stop_progress >= self.stop_duration:
                self.stopping = False
                self.stop_progress = 0.0
            return

        if self.moving:
            self.move_progress += self.tick_interval
            if self.move_progress >= self.move_duration:
                self.move_progress = 1.0
                self.position = self.target_pos
                self.moving = False
            return

        if self.rotating:
            self.rotation_progress += self.tick_interval
            if self.rotation_progress >= self.rotation_duration:
                self.rotating = False
                self.direction = self.target_direction

                if self.current_deadline is None:
                    raise RuntimeError(f"[Robot {self.robot_id}] deadline이 설정되지 않아 전진을 수행할 수 없습니다.")
                remaining = self.current_deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"[Robot {self.robot_id}] 전진을 위한 남은 시간이 없습니다: deadline 경과, 현재 시각 {time.time()}, deadline {self.current_deadline}")
                
                self.move_duration = remaining
                self.move_progress = 0.0
                self.move_forward()
                return
            
        else:
            if self.current_command is None and self.command_queue:
                self.current_command = self.command_queue.pop(0)
            if self.current_command:
                self.execute_command(self.current_command)
                self.current_command = None


    # 로봇 위치 반환
    def get_position(self):
        if self.moving:
            current = np.array(self.start_pos)
            target = np.array(self.target_pos)
            ratio = min(self.move_progress / self.move_duration, 1.0)  # 비율 보정
            return (1 - ratio) * current + ratio * target
        else:
            return self.position

        
    # 로봇 방향 반환
    def get_direction(self):
        # 1) 각도 매핑 (라디안)
        angles = {
            "north":  math.pi/2,
            "east":    0,
            "south": -math.pi/2,
            "west":   math.pi
        }

        # 회전 중이면 시작→종료 각도 보간
        if self.rotating:
            # 현재 t (0→1)
            t = min(self.rotation_progress / self.rotation_duration, 1.0)

            # 시작/목표 인덱스
            dirs = ["north", "east", "south", "west"]
            start_idx = dirs.index(self.direction)
            delta = self.rotation_steps if self.rotation_dir=="right" else -self.rotation_steps
            target_idx = (start_idx + delta) % 4

            # 시작/끝 각도
            start_ang = angles[dirs[start_idx]]
            end_ang   = angles[dirs[target_idx]]

            # 시계/반시계 회전을 올바르게 보간
            diff = end_ang - start_ang
            # 두 방향이 반대(±π)일 때, shortest path 선택
            if abs(diff) > math.pi:
                diff -= math.copysign(2*math.pi, diff)
            ang = start_ang + t * diff

            # 최종 방향 벡터 (화살표는 y축 반전 기준)
            dx = math.cos(ang)
            dy = -math.sin(ang)
            return (dx, dy)

        # 회전 중이 아니면 정방향 벡터
        dir_vecs = {
            "north": (0, -1),
            "east":  (1,  0),
            "south": (0,  1),
            "west":  (-1, 0)
        }
        return dir_vecs[self.direction]


    def execute_command(self, command):
        """
        command: (cmd_str, deadline) 형태만 허용.
        deadline 이 None 이면 호출 자체가 불가능하므로,
        항상 (cmd_str, deadline) 튜플로 들어옵니다.
        """
        cmd_str, deadline = command
        self.current_deadline = deadline

        # deadline 확인 (안정장치)
        if deadline is None:
            raise RuntimeError(f"[Robot {self.robot_id}] deadline이 없어 명령을 실행할 수 없습니다: {cmd_str}")

        # T-extension 은 그대로 처리
        if cmd_str.startswith("T"):
            self.next_move_extension = float(cmd_str[1:])
            return

        # 이동 명령 Dxx
        if cmd_str.startswith("D"):
            self.move_forward()
            # 남은 시간(초)만큼만 움직이도록
            remain = deadline - time.time()
            if remain <= 0:
                raise TimeoutError(f"[Robot {self.robot_id}] 이동 deadline 경과: {cmd_str}")
            self.move_duration = remain
            self.move_progress = 0.0
            self.next_move_extension = 0.0
            return

        # 회전 명령 Rxx
        if cmd_str.startswith("R"):
            from math import copysign
            directions = ["north", "east", "south", "west"]
            angle = int(cmd_str[1:])
            steps = abs(angle) // 90
            self.rotation_steps = steps
            self.target_direction = directions[
                (directions.index(self.direction) + int(copysign(1, angle)) * steps) % 4
            ]
            remain = deadline - time.time()
            if remain <= 0:
                raise TimeoutError(f"[Robot {self.robot_id}] 회전 deadline 경과: {cmd_str}")
            self.rotation_duration = MOTION_DURATIONS["Rotate"] * steps
            self.rotation_progress = 0.0
            self.rotating = True
            return

        # 그 외 모든 명령에 대해 deadline 필수
        raise NotImplementedError(f"[Robot {self.robot_id}] 지원하지 않는 명령이거나 deadline이 잘못됨: {cmd_str}")
