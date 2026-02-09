import cv2

video_path = "Input_video.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print("FPS:", fps)
print("Total frames:", frame_count)
print("Duration (sec):", duration)

cap.release()
