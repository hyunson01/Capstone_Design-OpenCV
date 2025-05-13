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
                if c.startswith("distance="):
                    val = int(c.split("=")[1])
                    compressed += 'f' * val
                elif c == "R":
                    compressed += 'r'
                elif c == "L":
                    compressed += 'l'
                elif c == "R R":
                    compressed += 'rr'
                elif c == "L L":
                    compressed += 'll'
                elif c == "S":
                    compressed += 's'
                else:
                    print(f"[FakeMQTT] 경고: 알 수 없는 명령어 '{c}' 무시됨")
            print(f"[FakeMQTT] → {topic} 로 '{compressed}' 전송")
            self.publish(topic, compressed)


