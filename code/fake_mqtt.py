class FakeMQTTClient:
    def __init__(self):
        self.messages = []

    def publish(self, topic, message):
        print(f"[Publish] {topic}: {message}")
        self.messages.append((topic, message))

    def subscribe(self, topic_filter):
        # 여기선 필터링 안 하고 다 받게 한다
        pass

    def receive(self):
        if self.messages:
            return self.messages.pop(0)
        return None