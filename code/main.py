import sys
import os
import cv2
import numpy as np
import subprocess 
import math
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ICBS_PATH = os.path.join(CURRENT_DIR, '..', 'MAPF-ICBS', 'code')
sys.path.append(os.path.normpath(ICBS_PATH))


from grid import load_grid, GRID_FOLDER
from interface import grid_visual, slider_create, slider_value, draw_agent_points, draw_paths
from config import grid_row, grid_col, cell_size, camera_cfg, IP_address_, MQTT_TOPIC_COMMANDS_ , MQTT_PORT , NORTH_TAG_ID, CORRECTION_COEF, critical_dist 
from vision.visionsystem import VisionSystem 
from vision.camera import camera_open, Undistorter 
from cbs.agent import Agent
from cbs.pathfinder import PathFinder
from align import send_center_align, send_north_align 
from recieve_message import (
    start_sequence, set_tag_info_provider, set_alignment_pending, alignment_pending,
    check_center_alignment_ok, check_north_alignment_ok, check_all_completed, start_auto_sequence,
    check_direction_alignment_ok, alignment_angle,
    pause_robots, resume_robots      
)
from config import cell_size_cm

SELECTED_RIDS = set()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CTS_SCRIPT = os.path.join(SCRIPT_DIR, "command_transfer.py") #별도의 창으로 command_transfer 실행

# 메인 로직이 실행되기 전에 커맨드 전송 스크립트를 백그라운드로 시작
# sys.executable: 현재 사용 중인 파이썬 인터프리터 경로
subprocess.Popen([sys.executable, CTS_SCRIPT],creationflags=subprocess.CREATE_NEW_CONSOLE)
print(f"▶ command_transfer_encoderSelf.py 별도 콘솔에서 실행: {CTS_SCRIPT}")


# 브로커 정보
# main.py 상단에 USE_MQTT 정의
USE_MQTT = 1  # 0: 비사용, 1: 사용

if USE_MQTT:
    from recieve_message import init_mqtt_client
    client = init_mqtt_client()   # ← recieve_message의 '그' 클라이언트 단일 사용
else:
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

#근접 시 즉시 정지 기능
PROXIMITY_GUARD_ENABLED = True   # 끄려면 False
PROXIMITY_STOP_LATCH = set()     # 이미 proximity로 im_S 보낸 로봇 ID(int)

grid_array = np.zeros((grid_row, grid_col), dtype=np.uint8)
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
PRESET_IDS = []


def compute_visible_robot_ids(tag_info: dict) -> list[int]:
    """카메라에 잡힌 '로봇' 태그 ID를 정렬 리스트로 반환 (보드/NORTH_TAG_ID 제외)."""
    visible = []
    for tid, data in tag_info.items():
        # tid는 정수, 'On' 상태, 보드 태그는 제외
        if isinstance(tid, int) and data.get("status") == "On" and tid != NORTH_TAG_ID:
            visible.append(tid)
    visible.sort()
    return visible


def _get_tag_cm(tag_info: dict, rid: int):
    d = tag_info.get(rid, {})
    if d.get("status") == "On" and "corrected_center" in d:
        return d["corrected_center"]  # (X_cm, Y_cm)
    return None

def compute_pairwise_distances_cm(tag_info: dict, ids: list[int]):
    """ids 목록에서 보이는 태그들 간의 모든 쌍 거리를 cm로 반환"""
    pairs = []
    for i, a in enumerate(ids):
        pa = _get_tag_cm(tag_info, a)
        if not pa:
            continue
        for b in ids[i+1:]:
            pb = _get_tag_cm(tag_info, b)
            if not pb:
                continue
            dx = pa[0] - pb[0]
            dy = pa[1] - pb[1]
            dist = math.hypot(dx, dy)
            pairs.append(((a, b), dist))
    return pairs

def proximity_guard(tag_info: dict, ids: list[int], threshold_cm: float):
    """
    ids에 대해 임계거리 이하 쌍이 하나라도 연결된 '충돌 클러스터'를 찾고,
    그 클러스터에 속한 모든 로봇 집합(to_stop)과 트리거 페어 목록을 반환.
    """
    pairs = compute_pairwise_distances_cm(tag_info, ids)
    adj = {rid: set() for rid in ids}
    trigger_pairs = []
    for (a, b), dist in pairs:
        if dist <= threshold_cm:
            adj[a].add(b); adj[b].add(a)
            trigger_pairs.append(((a, b), dist))

    to_stop = set()
    visited = set()
    for rid in ids:
        if rid in visited:
            continue
        # DFS로 연결 성분(클러스터) 추출
        stack = [rid]
        comp = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            stack.extend(v for v in adj[u] if v not in visited)
        if len(comp) >= 2:        # 2대 이상 연결 → 충돌 클러스터
            to_stop.update(comp)

    return to_stop, trigger_pairs

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
    path: [(r0,c0), (r1,c1), ...]
    init_hd: 0=북,1=동,2=남,3=서
    반환: [{'command': 'Stay'|'L90'|'R90'|'T185'|'F10_modeA'}, ...]
    """
    cmds = []
    hd = init_hd

    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        # 0) 같은 좌표 → '대기'
        if r0 == r1 and c0 == c1:
            cmds.append({'command': 'Stay'})
            continue

        # 1) 목표 방향
        if   r1 < r0:  desired = 0  # 북
        elif c1 > c0:  desired = 1  # 동
        elif r1 > r0:  desired = 2  # 남
        else:          desired = 3  # 서

        # 2) 회전/이동 단일 명령
        diff = (desired - hd) % 4
        if diff == 0:
            # 회전 불필요 → 전진만
            cmds.append({'command': f'F{cell_size_cm:.1f}_modeA'})
        elif diff == 1:
            cmds.append({'command': 'R90'})
        elif diff == 2:
            cmds.append({'command': 'T185'})  # 180도 보정치
        else:  # diff == 3
            cmds.append({'command': 'L90'})

        # 3) 헤딩 갱신
        hd = desired

    return cmds


YAW_TO_NORTH_OFFSET_DEG = 0  # 필요시 -90 / +90 / 180 등으로 보정

def yaw_to_hd(yaw_deg: float, offset_deg: float = 0) -> int:
    """연속각(yaw_deg)을 90° 섹터로 양자화하여 hd(0~3)로 변환"""
    ang = (yaw_deg + offset_deg) % 360.0
    return int(((ang + 45.0) // 90.0) % 4)

def get_initial_hd(robot_id: int) -> int:
    data = tag_info.get(robot_id)
    if not data or data.get('status') != 'On':
        return 0
    
    # 화면 표시용 방향/오차 값 사용
    delta = data.get("heading_offset_deg")
    if delta is None:
        return 0

    # base_dir 추출
    yaw_deg = (data.get("yaw_front_deg", 0) + 360) % 360
    direction_angles = [90, 0, 270, 180]  # N=90, W=0, S=270, E=180
    diffs = [abs(((yaw_deg - a + 180) % 360) - 180) for a in direction_angles]
    min_idx = diffs.index(min(diffs))
    hd = [0, 3, 2, 1][min_idx]  # N=0, E=1, S=2, W=3 로 매핑

    return hd


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
        hd0 = get_initial_hd(agent.id)  # ▶ 각 로봇의 현재 바라보는 방향으로 초기화
        cmds = path_to_commands(raw_path, hd0)

        basic_cmds = []
        cur_hd = hd0  
        for cmd_obj in cmds:
            cmd = cmd_obj["command"]
            basic_cmds.append(cmd)

            # 헤딩 업데이트 (기본 헤딩만 유지)
            if cmd.startswith("R"):
                cur_hd = (cur_hd + 1) % 4
            elif cmd.startswith("L"):
                cur_hd = (cur_hd - 1) % 4
            elif cmd.startswith("T"):
                cur_hd = (cur_hd + 2) % 4

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
    start_sequence(cmd_map)


#정지 함수
def send_emergency_stop(client):
    print("!! Emergency Stop 명령 전송: 'S' to robots 1~4")
    for rid in range(1, 5):
        topic = f"robot/{rid}/cmd"
        client.publish(topic, "S")
        print(f"  → Published to {topic}")
        
#정지 해제 함수        
def send_release_all(client, ids):
    for rid in ids:
        client.publish(f"robot/{rid}/cmd", "RE")
        print(f"▶ [Robot_{rid}] RE 전송")

#즉시 모터 정지 함수
def immediate_stop(client, ids):
    """선택된 로봇(들)에게 즉시 정지 im_S 전송"""
    for rid in ids:
        client.publish(f"robot/{rid}/cmd", "im_S")
        print(f"🛑 [Robot_{rid}] 즉시정지(im_S) 전송")
        
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

        # 1) 프레임 처리
        visionOutput = vision.process_frame(frame, detect_params)
        if visionOutput is None:
            continue
        dyn = vision.get_obstacle_grid()
        if dyn is not None:
            # 현재 그리드를 "순수 비전 결과"로 쓰려면:
            grid_array = dyn.copy()
            # 만약 화면에 그리드 시각화가 필요하면 아래처럼 사용
            vis = grid_visual(grid_array.copy())
        else:
            # 아직 보드 lock 전 등, 비전 그리드가 없을 땐 기존 grid_array 유지
            vis = grid_visual(grid_array.copy())

        # 2) 새 프레임 기반으로 화면/태그 정보 먼저 갱신
        frame = visionOutput["frame"]
        tag_info = visionOutput["tag_info"]

        # 3) 새 tag_info로 PRESET_IDS 갱신 (리스트 객체 유지)
        _prev = PRESET_IDS[:]                           # 이전 목록 백업
        new_ids = compute_visible_robot_ids(tag_info)   # 반드시 최신 tag_info 기반
        PRESET_IDS[:] = new_ids

        # 4) 변경 처리 + 합류 감지 시 'S'(키보드 t와 동일) 자동 전송
        if PRESET_IDS != _prev:
            # 숫자키로 선택해둔 로봇 중, 화면에 없는 애는 해제
            SELECTED_RIDS.intersection_update(set(PRESET_IDS))
            print(f"🔄 PRESET_IDS 갱신 → {PRESET_IDS}")

            # 새 로봇 합류(길이 증가) → 기존 로봇들 일시정지
            if len(PRESET_IDS) > len(_prev):
                joined = sorted(set(PRESET_IDS) - set(_prev))     # 새로 들어온 로봇
                to_pause = sorted(set(_prev) & set(PRESET_IDS))   # 기존(아직 보이는) 로봇
                if to_pause:
                    pause_robots([str(r) for r in to_pause])      # 현재 명령 완료 후 정지(S)
                    print(f"⏸ 합류 감지 {joined} → 기존 {to_pause}에 'S' 전송")
                    
        # 5) 근접 보호(critical_dist): 임계거리 이내 로봇들 즉시정지
        if PROXIMITY_GUARD_ENABLED and PRESET_IDS:
            # 화면에서 사라진 로봇은 래치에서도 제거
            PROXIMITY_STOP_LATCH.intersection_update(set(PRESET_IDS))

            to_stop, trigger_pairs = proximity_guard(tag_info, PRESET_IDS, critical_dist)

            # 새롭게 정지시킬 대상만 선별
            new_targets = [rid for rid in to_stop if rid not in PROXIMITY_STOP_LATCH]
            if new_targets:
                # 어떤 쌍들이 임계 이하였는지 로그
                for ((a, b), dist) in trigger_pairs:
                    print(f"⚠️ 근접 감지: ({a},{b}) 거리 = {dist:.2f} cm (기준 {critical_dist} cm)")

                immediate_stop(client, new_targets)
                PROXIMITY_STOP_LATCH.update(new_targets)
                print(f"🛑 근접 보호 작동 → 즉시정지 전송 대상: {sorted(new_targets)}")



        # 6) 최신 tag_info 공급자 등록 및 그리드 렌더링
        set_tag_info_provider(lambda: tag_info)
        vis = grid_visual(grid_array.copy())

        # 7) 태그 상태 출력/보조 처리
        for tag_id in [1, 2, 3, 4]:
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
        elif key == ord('m'):
            new_mode = 'contour' if vision.board_mode == 'tag' else 'tag'
            vision.set_board_mode(new_mode)
            print(f"Board mode switched to: {new_mode}")
            
        elif key == ord('c'):
            send_release_all(client, PRESET_IDS)
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
            saved = None
            if vision.obstacle_detector is not None and vision.obstacle_detector.last_occupancy is not None:
                # 날짜 기반 파일명(0828grid.json 등)
                saved = vision.obstacle_detector.save_grid(save_dir=GRID_FOLDER)  # 또는 "grid"
            print(f"Saved: {saved}" if saved else "No grid to save yet")
        
        #기존 북쪽정렬 
        elif key == ord('x'):
            send_release_all(client, PRESET_IDS)
            unaligned = [rid for rid in PRESET_IDS if not check_north_alignment_ok(str(rid))]
            for tag_id in unaligned:
                set_alignment_pending(str(tag_id), "north")
            if unaligned:
                send_north_align(client, tag_info, MQTT_TOPIC_COMMANDS_, NORTH_TAG_ID,
                                targets=unaligned, alignment_pending=alignment_pending)
                
        elif key == ord('f'):
            # 가장 가까운 동/서/남/북으로 정렬
            from align import send_direction_align
            unaligned = []
            for rid in PRESET_IDS:
                rid_str = str(rid)
                data = tag_info.get(rid, {})
                delta = data.get("heading_offset_deg", None)
                if delta is None or abs(delta) >=  alignment_angle:  # recieve_message.py의 동일 기준 사용
                    unaligned.append(rid)

            for tag_id in unaligned:
                set_alignment_pending(str(tag_id), "direction")

            if unaligned:
                send_direction_align(client, tag_info, MQTT_TOPIC_COMMANDS_,
                                    targets=unaligned, alignment_pending=alignment_pending)
            else:
                print("✅ 모든 대상이 이미 방향정렬 완료 상태")

        elif key == ord('a'):
            send_release_all(client, PRESET_IDS)
            unaligned = [rid for rid in PRESET_IDS if not check_center_alignment_ok(str(rid))]
            for tag_id in unaligned:
                set_alignment_pending(str(tag_id), "center")  # ✅ 먼저 pending 등록
            if unaligned:
                send_center_align(client, tag_info, MQTT_TOPIC_COMMANDS_, targets=unaligned, 
                                  alignment_pending=alignment_pending)
                
        # 숫자키로 대상 선택/토글 (예: 1~4)
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            rid = int(chr(key))
            if rid in SELECTED_RIDS:
                SELECTED_RIDS.remove(rid)
                print(f"[-] 선택 해제: {rid} / 현재 선택: {sorted(SELECTED_RIDS)}")
            else:
                SELECTED_RIDS.add(rid)
                print(f"[+] 선택 추가: {rid} / 현재 선택: {sorted(SELECTED_RIDS)}")

        # 선택 로봇 정지 (그냥 누르면 전체 정지)
        elif key == ord('t'):
            if SELECTED_RIDS:
                pause_robots([str(r) for r in SELECTED_RIDS])
            else:
                if PRESET_IDS:
                    pause_robots([str(r) for r in PRESET_IDS])
                    print(f"⏸ 모든 접속 로봇 정지 예약(S): {PRESET_IDS}")
                else:
                    print("⚠️ 정지할 접속 로봇이 없습니다.")

        # 선택 로봇 재개 (그냥 누르면 전체 정지)
        elif key == ord('y'):
            # 선택 대상을 RE로 재개, 없으면 전체 재개
            targets = sorted(SELECTED_RIDS) if SELECTED_RIDS else list(PRESET_IDS)
            if targets:
                send_release_all(client, targets)  # ← RE 전송
                # 래치 해제
                for r in targets:
                    PROXIMITY_STOP_LATCH.discard(int(r))
                print(f"▶ 재개(RE) 전송: {targets}")
            else:
                print("⚠️ 재개할 대상이 없습니다.")




        elif key == ord('d'):  # 자동 시퀀스 : 중앙정렬 -> 방향정렬 -> 경로진행 
            send_release_all(client, PRESET_IDS)

            start_auto_sequence(
                client, tag_info, PRESET_IDS, agents, MQTT_TOPIC_COMMANDS_, NORTH_TAG_ID,
                set_alignment_pending, alignment_pending,
                check_center_alignment_ok,          # 1단계: 중앙정렬 판정
                check_direction_alignment_ok,       # 2단계: 방향정렬 판정
                send_center_align,                  # 4단계: 마무리 중앙정렬 전송
                compute_cbs,
                check_all_completed
            )
            
        elif key in (ord('u'), ord('U')):  # 숫자 선택 후 U → 선택 대상 즉시 정지
            if SELECTED_RIDS:
                immediate_stop(client, sorted(SELECTED_RIDS))
            else:
                # 선택이 없으면 현재 화면에 잡힌 모든 로봇 즉시 정지
                if PRESET_IDS:
                    immediate_stop(client, PRESET_IDS)
                    print(f"🛑 모든 접속 로봇 즉시 정지(im_S): {PRESET_IDS}")
                else:
                    print("⚠️ 즉시 정지할 접속 로봇이 없습니다.")



    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
