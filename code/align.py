import json
import numpy as np
from config import  IP_address_, MQTT_TOPIC_COMMANDS_ , MQTT_PORT , NORTH_TAG_ID

def send_center_align(client, tag_info, MQTT_TOPIC_COMMANDS_, targets=None,alignment_pending=None):
    """
    중앙 정렬 명령 전송 (회전 + 직진)
    → alignment_pending에 있는 로봇만 대상
    """
    if targets is None:
        targets = list(tag_info.keys())

    for tag_id in targets:
        rid_str = str(tag_id)

        # ✅ pending에 등록된 로봇만 처리
        if rid_str not in alignment_pending:
            print(f"⏩ Robot_{rid_str} 는 중앙정렬 대상 아님 → 건너뜀")
            continue

        data = tag_info.get(tag_id)
        if data is None or data.get('status') != 'On':
            print(f"   ✗ Robot_{rid_str} 상태 비정상 → 건너뜀")
            continue

        # 거리(cm), 상대각도(°)
        d = data.get('dist_cm', 0.0)
        ry = data.get('relative_angle_deg', 0.0)

        # 명령 생성
        rot_cmd = f"{'L' if ry < 0 else 'R'}{abs(ry):.1f}_modeOnly"
        mov_cmd = f"F{d:.1f}_modeC"

        payload = {
            "commands": [{
                "robot_id": rid_str,
                "command_count": 2,
                "command_set": [
                    {"command": rot_cmd},
                    {"command": mov_cmd}
                ]
            }]
        }

        print(f"▶ 중앙정렬 명령 전송: Robot_{rid_str} → {rot_cmd} + {mov_cmd}")
        client.publish(MQTT_TOPIC_COMMANDS_, json.dumps(payload, ensure_ascii=False))



    
#북쪽정렬
def send_north_align(client, tag_info, MQTT_TOPIC_COMMANDS_, NORTH_TAG_ID, targets=None,alignment_pending=None):
    """
    북쪽 정렬 명령 전송 (회전만)
    → alignment_pending에 있는 로봇만 대상
    """
    north = tag_info.get(NORTH_TAG_ID)
    if north is None or north.get('status') != 'On':
        print(f"   ✗ 북쪽 태그(ID={NORTH_TAG_ID}) 상태 비정상")
        return

    north_yaw = north.get('yaw_front_deg', None)
    if north_yaw is None:
        print("   ✗ 북쪽 태그 yaw_front_deg 정보 없음")
        return

    if targets is None:
        targets = list(tag_info.keys())

    for tag_id in targets:
        rid_str = str(tag_id)

        # ✅ pending에 등록된 로봇만 처리
        if rid_str not in alignment_pending:
            print(f"⏩ Robot_{rid_str} 는 북쪽정렬 대상 아님 → 건너뜀")
            continue

        if tag_id == NORTH_TAG_ID:
            continue

        data = tag_info.get(tag_id)
        if data is None or data.get('status') != 'On':
            print(f"   ✗ Robot_{rid_str} 상태 비정상 → 건너뜀")
            continue

        cur_yaw = data.get('yaw_front_deg', None)
        if cur_yaw is None:
            print(f"   ✗ Robot_{rid_str} yaw_front_deg 없음")
            continue

        # Δ 각도 계산 (–180° ~ +180°)
        delta = ((cur_yaw - north_yaw + 180) % 360) - 180
        rot_deg = round(abs(delta), 1)
        cmd_letter = 'L' if delta > 0 else 'R'
        cmd = f"{cmd_letter}{rot_deg}_modeOnly"

        payload = {
            "commands": [{
                "robot_id": rid_str,
                "command_count": 1,
                "command_set": [{"command": cmd}]
            }]
        }

        print(f"▶ 북쪽정렬 명령 전송: Robot_{rid_str} → Δ={delta:.1f}° → {cmd}")
        client.publish(MQTT_TOPIC_COMMANDS_, json.dumps(payload, ensure_ascii=False))



