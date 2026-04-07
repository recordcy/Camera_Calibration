import cv2
import numpy as np
import os

# =========================
# 사용자 설정
# =========================
CALIB_FILE = "data/calibration_result.npz"

# 이미지 보정용 입력
INPUT_IMAGE = "data/frames/frame_000.jpg"
OUTPUT_IMAGE = "data/undistorted_image.jpg"

# 영상 보정용 입력
INPUT_VIDEO = "data/chess.mp4"
OUTPUT_VIDEO = "data/undistorted_video.mp4"


def undistort_image():
    if not os.path.exists(CALIB_FILE):
        print(f"[ERROR] Calibration file not found: {CALIB_FILE}")
        return

    data = np.load(CALIB_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"[ERROR] Cannot load image: {INPUT_IMAGE}")
        return

    h, w = img.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        1,
        (w, h)
    )

    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted_cropped = undistorted[y:y+rh, x:x+rw]
    else:
        undistorted_cropped = undistorted

    # 비교용 좌우 합치기
    comparison = np.hstack((img, undistorted))

    cv2.imwrite(OUTPUT_IMAGE, comparison)
    cv2.imwrite("data/undistorted_cropped.jpg", undistorted_cropped)

    print(f"[INFO] Saved comparison image: {OUTPUT_IMAGE}")
    print("[INFO] Left = original, Right = undistorted")


def undistort_video():
    if not os.path.exists(CALIB_FILE):
        print(f"[ERROR] Calibration file not found: {CALIB_FILE}")
        return

    data = np.load(CALIB_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {INPUT_VIDEO}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width * 2, height))

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        1,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        comparison = np.hstack((frame, undistorted))
        out.write(comparison)

    cap.release()
    out.release()

    print(f"[INFO] Saved undistorted video: {OUTPUT_VIDEO}")
    print("[INFO] Left = original, Right = undistorted")


if __name__ == "__main__":
    undistort_image()
    undistort_video()