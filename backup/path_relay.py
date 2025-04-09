class PathRelay:
    def __init__(self):
        self.agents = []

    def set_agents(self, agents):
        """Agent 객체 리스트를 저장한다."""
        self.agents = agents

    def get_agent(self):
        """Agent 객체 리스트를 반환한다."""
        return self.agents

    def find_agent(self, id=None, start=None):
        """id 또는 start로 agent를 찾는다."""
        if id is not None:
            return next((a for a in self.agents if a.id == id), None)
        if start is not None:
            return next((a for a in self.agents if a.start == start), None)
        return None

