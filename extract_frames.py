import cv2
import os

VIDEO_PATH = "data/chess.mp4"
OUTPUT_DIR = "data/frames"
SAVE_EVERY_N_FRAMES = 15  # 15프레임마다 1장 저장

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAVE_EVERY_N_FRAMES == 0:
            filename = os.path.join(OUTPUT_DIR, f"frame_{saved_idx:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved: {filename}")
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] Done. Total saved frames: {saved_idx}")

if __name__ == "__main__":
    main()