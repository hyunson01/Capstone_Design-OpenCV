
import time
import json
import paho.mqtt.client as mqtt
from align import send_center_align, send_north_align 
from config import MQTT_TOPIC_COMMANDS_, MQTT_PORT, IP_address_ ,NORTH_TAG_ID
import threading

# 정렬 명령 간 딜레이 (카메라 프레임 처리 보장용) DEBUG on_message
alignment_delay_sec = 1.2

DONE_TOPIC = "robot/done"

robot_command_map = {}     # 전체 명령
robot_indices = {}         # 현재 인덱스
last_heading = {}          # 로봇별 기준 방향 (0=북,1=동,2=남,3=서)
tag_info_provider = None   # 최신 yaw 값을 가져오는 콜백
active = False             # 실행 중 여부

# 정렬 반복 관련
alignment_pending = {}
alignment_angle= 1  # 임계각도 = 1도
alignment_dist = 1  # 임계거리 = 3cm

def set_alignment_pending(robot_id, mode):
    alignment_pending[robot_id] = {
        "mode": mode,
        "in_progress": False
    }
    print(f"▶ pending: {mode} <- Robot_{robot_id}")

def clear_alignment_pending(robot_id):
    if robot_id in alignment_pending:
        del alignment_pending[robot_id]

def check_center_alignment_ok(robot_id, dist_thresh=alignment_dist):
    tag_info = tag_info_provider()
    data = tag_info.get(int(robot_id))
    if not data or data.get("status") != "On":
        print(f"⚠️ Robot_{robot_id} 정렬용 태그 정보 없음 또는 비활성")
        return False

    dist = data.get("dist_cm", 0)
    print(f"[중앙정렬 거리 확인] Robot_{robot_id}: dist={dist:.2f} cm (기준: {dist_thresh} cm)")
    return abs(dist) <= dist_thresh

def check_north_alignment_ok(robot_id):
    if tag_info_provider is None:
        return False

    tag_info = tag_info_provider()
    tag = tag_info.get(int(robot_id))
    north = tag_info.get(NORTH_TAG_ID)

    if not tag or tag.get("status") != "On":
        print(f"⚠️ Robot_{robot_id} 태그 상태 비정상")
        return False
    if not north or north.get("status") != "On":
        print(f"⚠️ NORTH_TAG_ID={NORTH_TAG_ID} 상태 비정상")
        return False

    cur_yaw = tag.get("yaw_front_deg", None)
    north_yaw = north.get("yaw_front_deg", None)

    if cur_yaw is None or north_yaw is None:
        print(f"⚠️ yaw_front_deg 정보 없음: Robot_{robot_id} 또는 NORTH_TAG")
        return False

    # ✅ Δ 계산 (북쪽 기준 회전 오차)
    delta = ((cur_yaw - north_yaw + 180) % 360) - 180
    print(f"▶ Robot_{robot_id} Δ={delta:.2f}°, 기준: {alignment_angle:.1f}°")
    return abs(delta) < alignment_angle

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

    #print(f"DEBUG send_next_command: robot_id={robot_id!r}, idx={idx}, total_cmds={len(commands)}")
    if idx < len(commands):
        cmd = commands[idx]

        # 실시간 회전 보정
        if cmd.startswith(("R", "L")) and tag_info_provider:
            tag_info = tag_info_provider()
            tag = tag_info.get(int(robot_id))
            hd = last_heading.get(robot_id, 0)
            
            if tag and "yaw_front_deg" in tag:
                delta = tag.get("heading_offset_deg", 0)

                base_angle = 90
                if (delta > 0 and cmd.startswith("R")) or (delta < 0 and cmd.startswith("L")):
                    corrected_angle = base_angle - abs(delta)
                else:
                    corrected_angle = base_angle + abs(delta)

                corrected_angle = max(0, round(corrected_angle, 1))
                corrected_cmd = f"{cmd[0]}{corrected_angle}"
                cmd = corrected_cmd

                if cmd.startswith("R"):
                    hd = (hd + 1) % 4
                elif cmd.startswith("L"):
                    hd = (hd - 1) % 4
                last_heading[robot_id] = hd

        elif cmd.startswith("T"):
            hd = last_heading.get(robot_id, 0)
            hd = (hd + 2) % 4
            last_heading[robot_id] = hd

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
    #print(f"📩 DEBUG on_message: topic={msg.topic}, payload={msg.payload.decode()!r}")
    global active
    payload = msg.payload.decode()

    if payload.startswith("DONE;Robot_"):
        parts = payload.split(";")
        robot_id = parts[1].split("_")[1]
        cmd_info = parts[2] + (f";{parts[3]}" if len(parts) > 3 else "")

        print(f"✅ [Robot_{robot_id}] 명령 ({cmd_info}) 완료")
        #print(f"[정렬용 수신] {payload}")

        # ✅ 정렬 반복 처리 (modeOnly 명령)
        if "mode=modeOnly" in payload and robot_id in alignment_pending:
            info = alignment_pending[robot_id]
            mode = info["mode"]
            in_progress = info.get("in_progress", False)

            if mode == "north":
                if check_north_alignment_ok(robot_id):
                    print(f"✅ 북쪽 정렬 완료: Robot_{robot_id}")
                    clear_alignment_pending(robot_id)
                    return

                if in_progress:
                    print(f"⚠️ 이미 반복 중 → 건너뜀: Robot_{robot_id}")
                    return

                alignment_pending[robot_id]["in_progress"] = True

                def repeat_wrapper():
                    time.sleep(alignment_delay_sec)
                    if check_north_alignment_ok(robot_id):
                        print(f"✅ 북쪽 정렬 완료 (지연 후 재확인): Robot_{robot_id}")
                        clear_alignment_pending(robot_id)
                        if all(info["mode"] != "north" for info in alignment_pending.values()):
                            print("✅ 모든 로봇 북쪽정렬 완료")
                        return

                    tag_info = tag_info_provider()
                    send_north_align(client, tag_info, MQTT_TOPIC_COMMANDS_, NORTH_TAG_ID, targets=[int(robot_id)], alignment_pending=alignment_pending)
                    if robot_id in alignment_pending:
                        alignment_pending[robot_id]["in_progress"] = False

                threading.Thread(target=repeat_wrapper, daemon=True).start()

        if robot_id in alignment_pending:
            info = alignment_pending[robot_id]
            mode = info["mode"]
            in_progress = info.get("in_progress", False) 
            if mode == "center" and "cmd=MOVE" in payload:
                # ✅ 바뀐 핵심 조건: MOVE 명령이면 중앙정렬 평가 시작
                    print(f"📍 중앙정렬 직진 완료 메시지 감지: {payload}")

                    if check_center_alignment_ok(robot_id):
                        print(f"✅ 중앙정렬 완료: Robot_{robot_id}")
                        clear_alignment_pending(robot_id)
                        
                        if all(info["mode"] != "center" for info in alignment_pending.values()):
                            print("✅ 모든 로봇 중앙정렬 완료")
                        return
                    

                    if in_progress:
                        print(f"⚠️ 이미 반복 중 → 건너뜀: Robot_{robot_id}")
                        return

                    alignment_pending[robot_id]["in_progress"] = True

                    def repeat_wrapper_center():
                        print("🔁 중앙정렬 재시도 시작 (0.8초 대기 후)")
                        time.sleep(alignment_delay_sec)
                        
                        if check_center_alignment_ok(robot_id):  
                            print(f"✅ 중앙정렬 완료 (지연 후 재확인): Robot_{robot_id}")
                            clear_alignment_pending(robot_id)
                            return
                        tag_info = tag_info_provider()
                        send_center_align(client, tag_info, MQTT_TOPIC_COMMANDS_, targets=[int(robot_id)], alignment_pending=alignment_pending)

                        if robot_id in alignment_pending:
                            alignment_pending[robot_id]["in_progress"] = False

                    threading.Thread(target=repeat_wrapper_center, daemon=True).start()



        # ✅ 일반 명령 처리
        if active and robot_id in robot_command_map:
            send_next_command(robot_id)
            if check_all_completed():
                print("\n✅ [모든 명령 전송 완료]")
                active = False



def on_connect(client, userdata, flags, rc):
    client.subscribe(DONE_TOPIC)

client = None  # 전역으로 정의

def init_mqtt_client():
    global client
    import paho.mqtt.client as mqtt
    client = mqtt.Client()
    client.connect(IP_address_, MQTT_PORT, 60)
    client.loop_start()

def start_sequence(cmd_map):
    global robot_command_map, active, last_heading
    robot_command_map = cmd_map
    reset_all_indices()
    last_heading = {rid: 0 for rid in robot_command_map}
    active = True
    for rid in robot_command_map:
        send_next_command(rid)


def start_auto_sequence(client, tag_info, PRESET_IDS, agents, MQTT_TOPIC_COMMANDS_, NORTH_TAG_ID,
                        set_alignment_pending, alignment_pending,
                        check_center_alignment_ok, check_north_alignment_ok,
                        send_center_align, send_north_align,
                        compute_cbs, check_all_completed):
    
    def run_sequence():
        print("▶ 자동 시퀀스 시작: 중앙정렬 → 북쪽정렬 → 경로수행 → 재중앙정렬")

        # 1. 중앙정렬
        for rid in PRESET_IDS:
            rid_str = str(rid)
            if not check_center_alignment_ok(rid_str):
                print(f"📤 중앙정렬 명령 전송: Robot_{rid}")
                set_alignment_pending(rid_str, "center")
                send_center_align(client, tag_info, MQTT_TOPIC_COMMANDS_, targets=[int(rid)], alignment_pending=alignment_pending)
            else:
                print(f"✅ Robot_{rid} 중앙정렬 이미 완료")

        while any(str(rid) in alignment_pending and alignment_pending[str(rid)]["mode"] == "center" for rid in PRESET_IDS):
            time.sleep(0.2)
        print("✅ [1/4] 모든 로봇 중앙정렬 완료")

        # 2. 북쪽정렬
        for rid in PRESET_IDS:
            rid_str = str(rid)
            if not check_north_alignment_ok(rid_str):
                print(f"📤 북쪽정렬 명령 전송: Robot_{rid}")
                set_alignment_pending(rid_str, "north")
                send_north_align(client, tag_info, MQTT_TOPIC_COMMANDS_, NORTH_TAG_ID, targets=[int(rid)], alignment_pending=alignment_pending)
            else:
                print(f"✅ Robot_{rid} 북쪽정렬 이미 완료")

        while any(str(rid) in alignment_pending and alignment_pending[str(rid)]["mode"] == "north" for rid in PRESET_IDS):
            time.sleep(0.2)
        print("✅ [2/4] 모든 로봇 북쪽정렬 완료")

        # 3. 경로 수행
        if all(a.start and a.goal for a in agents):
            print("🚀 [3/4] 경로 수행 시작")
            compute_cbs()
        else:
            print("⚠️ 시작 또는 도착 지정되지 않은 에이전트 있음 → 중단")
            return

        while not check_all_completed():
            time.sleep(0.5)
        print("✅ [3/4] 모든 명령 완료")

        # 4. 마무리 중앙정렬
        for rid in PRESET_IDS:
            rid_str = str(rid)
            if not check_center_alignment_ok(rid_str):
                print(f"📤 (마무리) 중앙정렬 명령 전송: Robot_{rid}")
                set_alignment_pending(rid_str, "center")
                send_center_align(client, tag_info, MQTT_TOPIC_COMMANDS_, targets=[int(rid)], alignment_pending=alignment_pending)
            else:
                print(f"✅ Robot_{rid} 중앙정렬 이미 완료 (마무리 단계)")

        while any(str(rid) in alignment_pending and alignment_pending[str(rid)]["mode"] == "center" for rid in PRESET_IDS):
            time.sleep(0.2)

        print("✅ [4/4] 마무리 중앙정렬 완료")

    threading.Thread(target=run_sequence, daemon=True).start()
