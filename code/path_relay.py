_ids = None
_starts = None
_paths = None
_agents = None

def set_paths(ids, starts, paths):
    """CBSManager에서 id, start, path raw 데이터를 저장"""
    global _ids, _starts, _paths, _agents
    _ids = ids
    _starts = starts
    _paths = paths
    _agents = []
    for id_val, start_val, path_val in zip(ids, starts, paths):
        _agents.append({
            "id": id_val,
            "start": start_val,
            "path": path_val
        })

def get_paths():
    """전체 raw 데이터(id, start, path)를 반환"""
    return _ids, _starts, _paths

def get_agent():
    """전체 agents 딕셔너리를 반환"""
    return _agents

def find_agent(id=None, start=None):
    """id 또는 start로 agent 정보를 딕셔너리 통째로 반환"""
    if _agents is None:
        return None

    if id is not None:
        for agent in _agents:
            if agent["id"] == id:
                return agent  # 딕셔너리 통째로 반환

    if start is not None:
        for agent in _agents:
            if agent["start"] == start:
                return agent  # 딕셔너리 통째로 반환

    return None
