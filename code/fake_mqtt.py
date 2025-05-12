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
            compressed = "".join([
                'f' * int(cmd["command"].split('=')[1]) if "distance" in cmd["command"]
                else 'r' if cmd["command"] == "R"
                else 'l' if cmd["command"] == "L"
                else ''  # 기타 명령 무시
                for cmd in cs.to_dict()["command_set"]
            ])
            print(f"[FakeMQTT] → {topic} 로 '{compressed}' 전송")
            self.publish(topic, compressed)


