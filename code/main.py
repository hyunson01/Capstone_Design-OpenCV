import cv2
import numpy as np
from camera import camera_open, frame_process
from board import board_detect, perspective_transform, board_pts, board_origin, board_draw
from apriltag import AprilTagDetector, cm_per_px
from tracking import TrackingManager
from grid import save_grid, load_grid
from visual import mouse_callback, grid_visual, grid_tag_visual, info_tag, slider_create
from config import tag_info, object_points, camera_matrix, dist_coeffs,cbs_path, arguments
from trans import  run_cbs_manager, generate_movement_commands

def main():
    cap, fps = camera_open()
    frame_count = 0
    
    grid_array = load_grid()

    slider_create()
    
    tracking_manager = TrackingManager(window_size=5)
    tag_detector = AprilTagDetector()
    
    cv2.namedWindow("Grid Visualization")
    cv2.setMouseCallback("Grid Visualization", mouse_callback, param=grid_array)

    while True:
        frame_count += 1
        time = frame_count / fps
        frame, gray = frame_process(cap, camera_matrix, dist_coeffs)

        if frame is None:
            continue

        largest_rect = board_detect(gray)

        if largest_rect is not None:
            board_draw(frame, largest_rect)
            rect, board_width_px, board_height_px = board_pts(largest_rect)
            warped, warped_board_width_px, warped_board_height_px, warped_resized = perspective_transform(frame, rect, board_width_px, board_height_px)
            board_origin_tvec = board_origin(frame, rect[0])

            cm_per_pixel = cm_per_px(warped_board_width_px, warped_board_height_px)
            
            tags = tag_detector.tag_detect(gray)
            tag_detector.tags_process(tags, object_points, frame_count, board_origin_tvec, cm_per_pixel, frame, camera_matrix, dist_coeffs)
            tracking_manager.update_all(tag_info, time)
                
            info_tag(frame, tag_info)
            
            grid_visualization = grid_visual(grid_array)
            grid_tag_visual(grid_visualization, tag_info)
            
            cv2.imshow("Warped Perspective", warped_resized)
            cv2.imshow("Grid Visualization", grid_visualization)

            
        cv2.imshow("Detected Rectangle", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):  # 'q' 키 -> 종료 (저장 없이)
            break
        elif key == ord('s'):
            save_grid(grid_array)
        elif key == ord('c'):
            agents = run_cbs_manager(grid_array, tag_info)
            generate_movement_commands(agents)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
