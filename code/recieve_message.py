import time
import json
import paho.mqtt.client as mqtt
from config import MQTT_TOPIC_COMMANDS_, MQTT_PORT, IP_address_

DONE_TOPIC = "robot/done"

robot_command_map = {}     # 전체 명령
robot_indices = {}         # 현재 인덱스
last_heading = {}          # 로봇별 기준 방향 (0=북,1=동,2=남,3=서)
tag_info_provider = None   # 최신 yaw 값을 가져오는 콜백
active = False             # 실행 중 여부

class CommandSet:
    def __init__(self, robot_id, commands):
        self.robot_id = robot_id
        self.commands = commands

    def to_dict(self):
        return {
            "robot_id": self.robot_id,
            "command_count": len(self.commands),
            "command_set": [{"command": c} for c in self.commands]
        }

def set_tag_info_provider(func):
    global tag_info_provider
    tag_info_provider = func

def reset_all_indices():
    for rid in robot_command_map:
        robot_indices[rid] = 0

def send_next_command(robot_id):
    idx = robot_indices[robot_id]
    commands = robot_command_map[robot_id]

    print(f"DEBUG send_next_command: robot_id={robot_id!r}, idx={idx}, total_cmds={len(commands)}")
    if idx < len(commands):
        cmd = commands[idx]

        # 실시간 회전 보정
        if cmd.startswith(("R", "L")) and tag_info_provider:
            tag_info = tag_info_provider()
            tag = tag_info.get(int(robot_id))
            hd = last_heading.get(robot_id, 0)
            
            if tag and "yaw_front_deg" in tag:
                delta = tag.get("heading_offset_deg", 0)

                # 보정 계산
                base_angle = 90
                if (delta > 0 and cmd.startswith("R")) or (delta < 0 and cmd.startswith("L")):
                    corrected_angle = base_angle - abs(delta)
                else:
                    corrected_angle = base_angle + abs(delta)

                corrected_angle = max(0, round(corrected_angle, 1))
                corrected_cmd = f"{cmd[0]}{corrected_angle}"

                #print(f"📐 [Robot_{robot_id}] yaw={yaw:.1f}°, 기준={heading_deg}° → Δ={delta:+.1f}° → {cmd} → {corrected_cmd}")
                cmd = corrected_cmd

                # 헤딩 갱신
                if cmd.startswith("R"):
                    hd = (hd + 1) % 4
                elif cmd.startswith("L"):
                    hd = (hd - 1) % 4
                last_heading[robot_id] = hd

        elif cmd.startswith("T"):
            hd = last_heading.get(robot_id, 0)
            hd = (hd + 2) % 4
            last_heading[robot_id] = hd

        # 전송
        cs = CommandSet(robot_id, [cmd])
        payload = json.dumps({"commands": [cs.to_dict()]})
        print(f"📤 [Robot_{robot_id}] → {cmd}")
        client.publish(MQTT_TOPIC_COMMANDS_, payload)
        robot_indices[robot_id] += 1
    else:
        print(f"✅ [Robot_{robot_id}] 모든 명령 완료")


def check_all_completed():
    for rid in robot_command_map:
        if robot_indices[rid] < len(robot_command_map[rid]):
            return False
    return True

def on_message(client, userdata, msg):
    print(f"📩 DEBUG on_message: topic={msg.topic}, payload={msg.payload.decode()!r}")
    global active
    if not active:
        return
    payload = msg.payload.decode()
    if payload.startswith("DONE;Robot_"):
        parts = payload.split(";")
        robot_id = parts[1].split("_")[1]
        cmd_info = parts[2] + (f";{parts[3]}" if len(parts) > 3 else "")
        print(f"✅ [Robot_{robot_id}] 명령 ({cmd_info}) 완료")
        if robot_id in robot_command_map:
            send_next_command(robot_id)
        if check_all_completed():
            print("\n✅ [모든 명령 전송 완료]")
            active = False

def on_connect(client, userdata, flags, rc):
    client.subscribe(DONE_TOPIC)

client = mqtt.Client("SeqCmdSender")
client.on_connect = on_connect
client.on_message = on_message
client.connect(IP_address_, MQTT_PORT, 60)
client.loop_start()

def start_sequence(cmd_map):
    global robot_command_map, active, last_heading
    robot_command_map = cmd_map
    reset_all_indices()
    last_heading = {rid: 0 for rid in robot_command_map}  # 초기방향: 북쪽
    active = True
    for rid in robot_command_map:
        send_next_command(rid)
