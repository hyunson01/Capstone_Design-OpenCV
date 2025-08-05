import json
import numpy as np
from config import  IP_address_, MQTT_TOPIC_COMMANDS_ , MQTT_PORT , NORTH_TAG_ID

#그리드 중앙 정렬
def send_center_align(client, tag_info, MQTT_TOPIC_COMMANDS_):
    for tag_id, data in tag_info.items():
        if data.get('status') != 'On': continue
        # 거리(cm)·상대 각도(°)
        d  = data.get('dist_cm', 0.0)
        ry = data.get('relative_angle_deg', 0.0)
        # 회전·이동 명령 생성 (✅ 회전 전용으로 수정)
        rot_cmd = f"{'L' if ry<0 else 'R'}{abs(ry):.1f}_modeOnly"
        mov_cmd = f"F{d:.1f}_modeC"
        payload = {
            "commands": [{
                "robot_id":      str(tag_id),
                "command_count": 2,
                "command_set":   [
                    {"command": rot_cmd},
                    {"command": mov_cmd}
                ]
            }]
        }
        print("▶ Auto Alignment 명령 전송:", json.dumps(payload, ensure_ascii=False))
        client.publish(MQTT_TOPIC_COMMANDS_, json.dumps(payload, ensure_ascii=False))

    

#북쪽 정렬
def send_north_align(client, tag_info, MQTT_TOPIC_COMMANDS_, NORTH_TAG_ID):
    north = tag_info.get(NORTH_TAG_ID)
    if north is None or north.get('status') != 'On':
        print(f"   ✗ 북쪽 태그(ID={NORTH_TAG_ID}) 상태가 올바르지 않습니다.")
        return
    
    # ✅ 보정된 front 방향 기준 yaw 사용
    north_yaw = north.get('yaw_front_deg', None)
    if north_yaw is None:
        print("   ✗ 북쪽 태그의 yaw_front_deg 정보가 없습니다.")
        return

    for tag_id, data in tag_info.items():
        if data.get('status') != 'On' or tag_id == NORTH_TAG_ID:
            continue
        
        cur_yaw = data.get('yaw_front_deg', None)
        if cur_yaw is None:
            print(f"   ✗ ID={tag_id}에 yaw_front_deg 정보 없음")
            continue

        # Δ각도 계산 (–180°~+180° 정규화)
        delta = ((cur_yaw - north_yaw + 180) % 360) - 180
        rot_deg = round(abs(delta), 1)
        cmd_letter = 'L' if delta > 0 else 'R'
        cmd = f"{cmd_letter}{rot_deg}_modeOnly"  # ✅ 회전 전용 명령

        payload = {
            "commands": [{
                "robot_id":      str(tag_id),
                "command_count": 1,
                "command_set":   [{"command": cmd}]
            }]
        }
        print(f"   • ID={tag_id}: Δ={delta:.1f}° → 명령={cmd}")
        client.publish(MQTT_TOPIC_COMMANDS_, json.dumps(payload, ensure_ascii=False))
