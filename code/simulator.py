# simulator.py
import cv2
import numpy as np
from fake_mqtt import FakeMQTTBroker

class Simulator:
    def __init__(self, map_array, colors, cell_size=50):
        self.map_array = map_array
        self.colors = colors
        self.cell_size = cell_size
        self.robots = []
        self.vis = self.create_grid()
        self.paused = False

    def add_robot(self, robot_id, broker, start_pos=(0, 0)):
        robot = Robot(robot_id, broker, start_pos)
        self.robots.append(robot)
        print(f"Simulator: 로봇 {robot_id} 추가 완료. 시작 위치: {start_pos}")

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

    def run_once(self):
        if getattr(self, 'paused', False):
            return
        self.vis = self.create_grid()
        self.draw_robots()
        cv2.imshow("Simulator", self.vis)

    def draw_robots(self):
        for robot in self.robots:
            cx = int(robot.position[1] * self.cell_size + self.cell_size // 2)
            cy = int(robot.position[0] * self.cell_size + self.cell_size // 2)
            color = self.colors[robot.robot_id % len(self.colors)]
            cv2.circle(self.vis, (cx, cy), self.cell_size // 3, color, -1)

    def run(self):
        while True:
            self.run_once()
            key = cv2.waitKey(100)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()


class Robot:
    def __init__(self, robot_id, broker, start_pos):
        self.robot_id = robot_id
        self.broker = broker
        self.position = start_pos  # (row, col)
        self.direction = "north"   # 초기 방향
        self.broker.subscribe(f"robot/{self.robot_id}/move", self.on_receive_command)
        print(f"Robot {self.robot_id}: 구독 시작 (토픽: robot/{self.robot_id}/move)")

    def on_receive_command(self, command):
        print(f"[Robot {self.robot_id}] 받은 명령: {command}")
        self.execute_command(command)

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

    def move_forward(self):
        x, y = self.position
        if self.direction == "north":
            self.position = (x - 1, y)
        elif self.direction == "east":
            self.position = (x, y + 1)
        elif self.direction == "south":
            self.position = (x + 1, y)
        elif self.direction == "west":
            self.position = (x, y - 1)
        print(f"[Robot {self.robot_id}] 앞으로 이동 → 현재 위치: {self.position}")

    def turn_left(self):
        directions = ["north", "west", "south", "east"]
        idx = directions.index(self.direction)
        self.direction = directions[(idx + 1) % 4]
        print(f"[Robot {self.robot_id}] 왼쪽 회전 → 방향: {self.direction}")

    def turn_right(self):
        directions = ["north", "east", "south", "west"]
        idx = directions.index(self.direction)
        self.direction = directions[(idx + 1) % 4]
        print(f"[Robot {self.robot_id}] 오른쪽 회전 → 방향: {self.direction}")
