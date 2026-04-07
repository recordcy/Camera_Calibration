import cv2
import numpy as np
import os
import glob

# =========================
# 사용자 설정
# =========================
IMAGE_DIR = "data/frames"
IMAGE_EXT = "*.jpg"

# 체스보드 내부 코너 수
CHESSBOARD_SIZE = (7, 5)

SQUARE_SIZE = 25.0

# 코너 시각화 이미지 저장 폴더
DEBUG_DIR = "data/debug_corners"

# 결과 저장 파일
OUTPUT_FILE = "data/calibration_result.npz"


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # 종료 조건
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    # 3D 실제 좌표 생성
  
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, IMAGE_EXT)))

    if not image_paths:
        print(f"[ERROR] No images found in {IMAGE_DIR}")
        return

    image_size = None
    success_count = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] Failed to load image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체스보드 코너 찾기
        found, corners = cv2.findChessboardCorners(
            gray,
            CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        vis = img.copy()

        if found:
            # 서브픽셀 정밀화
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            objpoints.append(objp)
            imgpoints.append(corners2)
            success_count += 1

            cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners2, found)
            print(f"[OK] Corners detected: {path}")
        else:
            print(f"[FAIL] Corners not detected: {path}")

        debug_name = os.path.join(DEBUG_DIR, os.path.basename(path))
        cv2.imwrite(debug_name, vis)

        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

    if success_count < 5:
        print("[ERROR] Too few valid images for calibration. Try more frames or better views.")
        return

    # 캘리브레이션 수행
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    # RMSE 계산
    total_error = 0
    for i in range(len(objpoints)):
        projected_imgpoints, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
        error = cv2.norm(imgpoints[i], projected_imgpoints, cv2.NORM_L2) / len(projected_imgpoints)
        total_error += error

    mean_error = total_error / len(objpoints)

    # 결과 출력
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print("\n===== Calibration Result =====")
    print(f"Valid images: {success_count}")
    print(f"RMS returned by OpenCV: {ret}")
    print(f"Mean reprojection error (RMSE-like): {mean_error}")
    print("Camera Matrix:")
    print(camera_matrix)
    print("Distortion Coefficients:")
    print(dist_coeffs.ravel())
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")

    # 저장
    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rms=ret,
        mean_error=mean_error
    )

    print(f"\n[INFO] Calibration result saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()