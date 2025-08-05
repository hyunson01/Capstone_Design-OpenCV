import sys
import os
import cv2
import numpy as np
import subprocess 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ICBS_PATH = os.path.join(CURRENT_DIR, '..', 'MAPF-ICBS', 'code')
sys.path.append(os.path.normpath(ICBS_PATH))


from grid import load_grid
from interface import grid_visual, slider_create, slider_value, draw_agent_points, draw_paths
from config import grid_row, grid_col, cell_size, camera_cfg, IP_address_, MQTT_TOPIC_COMMANDS_ , MQTT_PORT , NORTH_TAG_ID, CORRECTION_COEF

from vision.visionsystem import VisionSystem 
from vision.camera import camera_open, Undistorter 
from cbs.agent import Agent
from cbs.pathfinder import PathFinder
# from recieve_message import start_sequence,set_tag_info_provider
from align import send_center_align, send_north_align # 중앙정렬, 북쪽정렬 함수


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CTS_SCRIPT = os.path.join(SCRIPT_DIR, "command_transfer.py") #별도의 창으로 command_transfer 실행

# 메인 로직이 실행되기 전에 커맨드 전송 스크립트를 백그라운드로 시작
# sys.executable: 현재 사용 중인 파이썬 인터프리터 경로
subprocess.Popen([sys.executable, CTS_SCRIPT],creationflags=subprocess.CREATE_NEW_CONSOLE)
print(f"▶ command_transfer_encoderSelf.py 별도 콘솔에서 실행: {CTS_SCRIPT}")


# 브로커 정보
# main.py 상단에 USE_MQTT 정의
USE_MQTT = 0  # 0: 비사용, 1: 사용

if USE_MQTT:
    import paho.mqtt.client as mqtt

    # 1) MQTT 클라이언트 생성
    client = mqtt.Client()

    # 2) 접속 (blocking call이 아니도록 loop_start 권장)
    client.connect(IP_address_, MQTT_PORT, 60)
    client.loop_start()
else:
    # Dummy 설정: publish 호출은 콘솔 출력으로 대체
    MQTT_TOPIC_COMMANDS_ = None

    class _DummyClient:
        def publish(self, topic, payload):
            print(f"[MQTT_DISABLED] publish → topic={topic}, payload={payload}")

    client = _DummyClient()

correction_coef_value = CORRECTION_COEF

def correction_trackbar_callback(val):
    global correction_coef_value
    correction_coef_value = val / 100.0
    print(f"[INFO] 실시간 보정계수: {correction_coef_value:.2f}")

cv2.namedWindow("CorrectionPanel", cv2.WINDOW_NORMAL)
cv2.createTrackbar(
    "Correction Coef", "CorrectionPanel",
    int(CORRECTION_COEF * 100), 200, correction_trackbar_callback
)
correction_trackbar_callback(int(CORRECTION_COEF * 100))  # 초기화

# 전역 변수
agents = []
paths = []
manager = None
pathfinder = None
grid_array = None
visualize = True
# tag_info 전역 변수 초기화
tag_info = {}

    # 비전 시스템 초기화
video_path = r"C:/img/test2.mp4"
cap, fps = camera_open(source=None)

undistorter = Undistorter(
    camera_cfg['type'],
    camera_cfg['matrix'],
    camera_cfg['dist'],
    camera_cfg['size']
)
vision = VisionSystem(undistorter=undistorter, visualize=True)
vision.correction_coef_getter = lambda: correction_coef_value

# 사용할 ID 목록
#PRESET_IDS = [1,2,3,4]  # 예시: 1~12까지의 ID 사용
PRESET_IDS = [1,3]

# 마우스 콜백 함수
def mouse_event(event, x, y, flags, param):
    """
    좌클릭  : 출발지(start) 지정
    우클릭  : 도착지(goal)  지정
    - PRESET_IDS(예: [2, 4]) 두 개가 모두 완성되면 CBS 실행
    """
    global agents, paths, pathfinder

    row, col = y // cell_size, x // cell_size
    if not (0 <= row < grid_row and 0 <= col < grid_col):
        return

    updated = False                 # ← 변경 여부 플래그
    complete_agents = [a for a in agents if a.start and a.goal]

    # ---------- 1. 출발지 클릭 ----------
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Start set at ({row}, {col})")

        # 1‑A. 이미 완성된 agent가 한도(PRESET_IDS)만큼이면 생성 제한
        if len(complete_agents) >= len(PRESET_IDS):
            print("더 이상 agent를 생성할 수 없습니다.")
            return
        
        if event == cv2.EVENT_LBUTTONDOWN and any(a.start == (row, col) for a in agents):
            return

        # 1‑B. goal‑only agent에 start 채우기
        for agent in agents:
            if agent.start is None and agent.goal is not None:
                agent.start = (row, col)
                updated = True
                break

        # 1‑C. start‑only agent의 start 덮어쓰기
        if not updated:
            for agent in agents:
                if agent.start is not None and agent.goal is None:
                    agent.start = (row, col)
                    updated = True
                    break

        # 1‑D. 둘 다 없으면 새 agent 생성
        if not updated:
            # 사용하지 않은 ID 선택
            used_ids = {a.id for a in agents}
            avail_ids = [pid for pid in PRESET_IDS if pid not in used_ids]
            if not avail_ids:
                print("더 이상 agent를 생성할 수 없습니다.")
                return
            new_id = avail_ids[0]
            agent = Agent(id=new_id, start=(row, col), goal=None, delay=0)
            agents.append(agent)
            updated = True

    # ---------- 2. 도착지 클릭 ----------
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Goal set at ({row}, {col})")

        # 2‑A. 이미 완성된 agent가 한도만큼이면 생성 제한
        if len(complete_agents) >= len(PRESET_IDS):
            print("더 이상 agent를 생성할 수 없습니다.")
            return

        # 2‑B. start‑only agent에 goal 채우기
        for agent in agents:
            if agent.goal is None and agent.start is not None:
                agent.goal = (row, col)
                updated = True
                break

        # 2‑C. goal‑only agent의 goal 덮어쓰기
        if not updated:
            for agent in agents:
                if agent.goal is not None and agent.start is None:
                    agent.goal = (row, col)
                    updated = True
                    break

        # 2‑D. 둘 다 없으면 새 agent 생성 (goal‑only)
        if not updated:
            used_ids = {a.id for a in agents}
            avail_ids = [pid for pid in PRESET_IDS if pid not in used_ids]
            if not avail_ids:
                print("더 이상 agent를 생성할 수 없습니다.")
                return
            new_id = avail_ids[0]
            agent = Agent(id=new_id, start=None, goal=(row, col), delay=0)
            agents.append(agent)
            updated = True

    # ---------- 3. 공통 후처리 ----------
    if updated:
        target_ids = set(PRESET_IDS)  # ← PRESET_IDS 기반으로 변경
        ready_ids  = {a.id for a in agents if a.start and a.goal and a.id in target_ids}

        if ready_ids == target_ids:
            print(f"Agent {sorted(ready_ids)} 준비 완료. CBS 실행.")
            pathfinder = compute_cbs(
                agents, paths, pathfinder, grid_row, grid_col, tag_info, path_to_commands
            )

            
# 태그를 통해 에이전트 업데이트
def update_agents_from_tags(tag_info):        # cm → 셀 좌표
    for tag_id, data in tag_info.items():
        if tag_id not in PRESET_IDS:
            continue
        if data.get("status") != "On":
            continue

        start_cell = data["grid_position"]                # (row, col)

        existing = next((a for a in agents if a.id == tag_id), None)
        if existing:                                      # 이미 agent 존재
            # ② 위치가 그대로면 아무것도 하지 않고 다음 tag로
            if existing.start == start_cell:
                continue
            existing.start = start_cell                   # 새 좌표로 갱신
        else:                                             # 처음 보는 tag
            agents.append(
                Agent(id=tag_id, start=start_cell, goal=None, delay=0)
            )



def path_to_commands(path, init_hd=0):
    """
    path: [(r0,c0), (r1,c1), ...] 그리드 좌표 리스트
    init_hd: 초기 헤딩 (0=북,1=동,2=남,3=서)
    반환: [{'command':'L90'|'R90'|'T180'|'F10_modeA'}, ...]
    """
    cmds = []
    hd = init_hd

    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        # 1) 목표 방향 계산
        if   r1 <  r0:
            desired = 0  # 북
        elif c1 >  c0:
            desired = 1  # 동
        elif r1 >  r0:
            desired = 2  # 남
        else:
            desired = 3  # 서

        # 2) 회전(diff) 처리 & 단일명령 생성
        diff = (desired - hd) % 4
        if   diff == 1:
            cmds.append({'command': 'R90'})
        elif diff == 2:
            cmds.append({'command': 'T180'})
        elif diff == 3:
            cmds.append({'command': 'L90'})
        else:  # diff == 0 → 순수 전진
            cmds.append({'command': 'F15_modeA'})

        # 3) 헤딩 갱신
        hd = desired

    return cmds

def compute_cbs():
    global paths, pathfinder, grid_array

    grid_array = load_grid(grid_row, grid_col)
    if pathfinder is None:
        pathfinder = PathFinder(grid_array)

    ready_agents = [a for a in agents if a.start and a.goal]
    if not ready_agents:
        print("⚠️  start·goal이 모두 지정된 에이전트를 찾을 수 없습니다.")
        return

    solved_agents = pathfinder.compute_paths(ready_agents)
    new_paths = [agent.get_final_path() for agent in solved_agents]
    if not new_paths:
        print("No solution found.")
        return

    paths.clear()
    paths.extend(new_paths)
    print("Paths updated via PathFinder.")

    # 🔁 보정 없이 원본 명령만 생성
    payload_commands = []
    for agent in solved_agents:
        raw_path = agent.get_final_path()
        hd = 0  # 초기 헤딩 (북쪽 기준)
        cmds = path_to_commands(raw_path, hd)

        basic_cmds = []
        for cmd_obj in cmds:
            cmd = cmd_obj["command"]
            basic_cmds.append(cmd)

            # 헤딩 업데이트 (기본 헤딩만 유지)
            if cmd.startswith("R"):
                hd = (hd + 1) % 4
            elif cmd.startswith("L"):
                hd = (hd - 1) % 4
            elif cmd.startswith("T"):
                hd = (hd + 2) % 4

        payload_commands.append({
            "robot_id": str(agent.id),
            "command_count": len(basic_cmds),
            "command_set": basic_cmds
        })

    # 전송용 딕셔너리
    cmd_map = {
        p["robot_id"]: p["command_set"]
        for p in payload_commands
    }

    print("▶ 순차 전송 시작:", cmd_map)
    # start_sequence(cmd_map)



def send_emergency_stop(client):
    print("!! Emergency Stop 명령 전송: 'S' to robots 1~4")
    for rid in range(1, 5):
        topic = f"robot/{rid}/cmd"
        client.publish(topic, "S")
        print(f"  → Published to {topic}")
    


def main():
    # 초기 설정
    global agents, paths, manager, visualize, tag_info

    # 그리드 불러오기
    base_grid = load_grid(grid_row, grid_col)
    grid_array = base_grid.copy()

    # 슬라이더 생성
    slider_create()
    detect_params = slider_value()  # 슬라이더에서 받아오기

    cv2.namedWindow("Video_display", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Video_display", vision.mouse_callback)
    cv2.namedWindow("CBS Grid")
    cv2.setMouseCallback("CBS Grid", mouse_event)

    while True:

        ret, frame = cap.read()
        if not ret:
            print("프레임 획득 실패")
            continue

        visionOutput = vision.process_frame(frame, detect_params)
        # set_tag_info_provider(lambda: tag_info)

        if visionOutput is None:
            continue
        vis = grid_visual(grid_array.copy())

        frame = visionOutput["frame"]
        # 전역 tag_info 변수에 업데이트
        tag_info = visionOutput["tag_info"]

        for tag_id in [1,2,3,4]:
            data = tag_info.get(tag_id)
            if data is None:
                continue

            status = data.get('status')
            if status != 'On':
                print(f"▶ Tag {tag_id}: 상태 = {status}")
                continue

            # 라디안 → 도 단위 변환
            yaw_rad = data.get('yaw', 0.0)
            yaw_deg = np.degrees(yaw_rad)

        if any("grid_position" in data for data in visionOutput["tag_info"].values()):
            update_agents_from_tags(visionOutput["tag_info"])

        # UI 시각화 화면
        draw_paths(vis, paths)
        draw_agent_points(vis, agents)

        cv2.imshow("CBS Grid", vis)
        cv2.imshow("Video_display", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):  # 'q' 키 -> 종료 (저장 없이)
            break
        elif key == ord('r'):
            print("Reset all")
            agents.clear()
            paths.clear()
            
        elif key == ord('c'):
            if all(a.start and a.goal for a in agents):
                compute_cbs()
            else:
                print("start 또는 goal이 비어 있는 에이전트가 있습니다.")
        elif key == ord('n'):
            vision.lock_board()
            print("보드 고정됨")
        elif key == ord('b'):
            vision.reset_board()
            print("🔄 고정된 보드를 해제")
        elif key == ord('v'):
            vision.toggle_visualization()
            print(f"시각화 모드: {'ON' if vision.visualize else 'OFF'}")
        elif key == ord('s'):
            vision.start_roi_selection()
        elif key == ord('x'):  # 북쪽정렬
            send_north_align(client, tag_info, MQTT_TOPIC_COMMANDS_, NORTH_TAG_ID)
        elif key == ord('a'):  # 중앙정렬
            send_center_align(client, tag_info, MQTT_TOPIC_COMMANDS_)
        elif key == ord('t'):  # 긴급정지
            send_emergency_stop(client)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
