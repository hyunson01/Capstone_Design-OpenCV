import time
import json
import paho.mqtt.client as mqtt

MQTT_SERVER = "192.168.159.132"
MQTT_PORT = 1883
TRANSFER_TOPIC = "command/transfer"

# 명령세트를 체계적으로 관리하기 위한 클래스 정의
class CommandSet:
    def __init__(self, robot_id, commands):
        """
        robot_id: 제어할 로봇의 ID (문자열)
        commands: 순서대로 실행할 개별 명령 리스트 (예: ["D30", "R", "D20", "R", "R", "D30", "L"])
        """
        self.robot_id = robot_id
        self.commands = commands
        self.command_count = len(commands)
    
    def convert_command(self, cmd):
        """
        간결한 형식(D숫자)을 "distance=숫자" 형식으로 변환합니다.
        예를 들어, "D30" -> "distance=30"
        그 외의 명령은 그대로 반환합니다.
        """
        if cmd.startswith("D"):
            # D 뒤에 있는 문자열을 숫자로 변환 시도 (정수 또는 소수)
            try:
                # float로 변환할 수 있으면 distance 명령으로 치환
                value = float(cmd[1:])
                # 소수점 이하가 0이면 정수 문자열로, 아니면 그대로
                if value.is_integer():
                    value = int(value)
                return f"distance={value}"
            except Exception as e:
                # 변환 오류 시 원본 문자열 사용
                return cmd
        # R, L 등은 그대로 반환
        return cmd

    def to_dict(self):
        """
        전체 명령세트를 딕셔너리 형태로 반환합니다.
        각 개별 명령에 대해 order(순서)와 command(명령)를 포함합니다.
        내부적으로 D숫자 형식은 "distance=" 형식으로 변환됩니다.
        """
        structured_commands = []
        for i, cmd in enumerate(self.commands, start=1):
            converted_cmd = self.convert_command(cmd)
            structured_commands.append({ "command": converted_cmd})
        return {
            "robot_id": self.robot_id,
            "command_count": self.command_count,
            "command_set": structured_commands
        }

# 로봇 2와 로봇 4의 명령세트 생성
command_set_robot2 = CommandSet("2", ["D30", "R", "D30", "R", "D30", "R", "D30"])
command_set_robot4 = CommandSet("4", ["D30", "R", "D30", "R", "D30", "R", "D30"])

# 여러 로봇의 명령세트를 하나의 메시지로 구성
message = {
    "commands": [
        command_set_robot2.to_dict(),
        command_set_robot4.to_dict()
    ]
}

# client = mqtt.Client()
# client.connect(MQTT_SERVER, MQTT_PORT, 60)

# payload = json.dumps(message)
# print("전송 모듈 명령 세트:", payload)
# client.publish(TRANSFER_TOPIC, payload)
# time.sleep(1)
# client.disconnect()
