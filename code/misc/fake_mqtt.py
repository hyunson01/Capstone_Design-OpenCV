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
                callback(message)  # 등록된 콜백을 호출
