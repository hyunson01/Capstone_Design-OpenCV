# fake_mqtt.py
class FakeMQTTBroker:
    def __init__(self):
        self.subscribers = dict()

    def subscribe(self, topic, callback):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def publish(self, topic, message):
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(message)

    def send_command_sets(self, command_sets):
        for cs in command_sets:
            topic = f"robot/{cs.robot_id}/move"
            compressed = ""
            for cmd in cs.to_dict()["command_set"]:
                c = cmd["command"]

                if c.startswith("D"):  # 예: D30 → fff
                    try:
                        val = int(c[1:])  # "D30" → 30
                        steps = val // 10  # 10cm 단위로 변환
                        compressed += 'f' * steps
                    except Exception:
                        print(f"[FakeMQTT] 경고: 잘못된 거리 명령 '{c}' 무시됨")
                
                elif c.startswith("R"):
                    try:
                        angle = int(c[1:])
                        if angle == 90:
                            compressed += 'r'
                        elif angle == -90:
                            compressed += 'l'
                        elif abs(angle) == 180:
                            compressed += 'rr'  # 180도는 오른쪽 두 번으로 처리
                        else:
                            print(f"[FakeMQTT] 경고: 지원되지 않는 회전 각도 '{angle}' 무시됨")
                    except Exception:
                        print(f"[FakeMQTT] 경고: 잘못된 회전 명령 '{c}' 무시됨")
                
                elif c == "S":
                    compressed += 's'
                
                else:
                    print(f"[FakeMQTT] 경고: 알 수 없는 명령어 '{c}' 무시됨")
            
            print(f"[FakeMQTT] → {topic} 로 '{compressed}' 전송")
            self.publish(topic, compressed)



