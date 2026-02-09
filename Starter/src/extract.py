import cv2
import os

def save_first_2_seconds_frames(video_path, output_dir="frames_2sec", duration_sec=2):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error: Cannot open video file with OpenCV")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Error: FPS not detected")

    total_frames_to_save = int(fps * duration_sec)

    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Saving first {duration_sec} seconds")
    print(f"[INFO] Total frames to save: {total_frames_to_save}")

    frame_index = 0

    while frame_index < total_frames_to_save:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended early")
            break

        frame_path = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_index += 1

    cap.release()
    print("[INFO] Done saving frames")


if __name__ == "__main__":
    video_path = "/Users/ashithrai/Documents/projects/RtOSE_06/Starter/video/Input_video2.mp4"
    save_first_2_seconds_frames(video_path)
