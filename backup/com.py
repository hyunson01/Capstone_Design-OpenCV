
while True:
    frame_count += 1

    if largest_rect is not None:
        
        
        #태그 관련
        for tag in tags:
   
        # 변환된 화면 및 배열 시각화 표시
        cv2.imshow("Warped Perspective", warped_resized)
        cv2.imshow("Grid Visualization", grid_visual)

    # 디스플레이 (실시간 출력)
    cv2.imshow("Detected Rectangle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()