# Camera_Calibration

## 📌 Overview
This project performs **camera calibration** and **lens distortion correction** using OpenCV.  
A chessboard pattern is used to estimate the intrinsic parameters of the camera and correct distortion in images and videos.

---

## ⚙️ Environment
- Python 3.x
- OpenCV (`cv2`)
- NumPy

---

## 📁 Project Structure
camera_calibration/
├── data/
│ ├── chess.mp4
│ ├── frames/
│ ├── debug_corners/
│ ├── calibration_result.npz
│ ├── undistorted_image.jpg
│ └── undistorted_video.mp4
├── extract_frames.py
├── camera_calibration.py
├── distortion_correction.py
└── README.md


---

## 🧠 Method

### 1. Frame Extraction
- Extract frames from a recorded chessboard video

### 2. Chessboard Corner Detection
- Detect internal corners of the chessboard
- Only valid frames are used for calibration

### 3. Camera Calibration
- Compute camera intrinsic matrix
- Estimate distortion coefficients

### 4. Distortion Correction
- Apply calibration results to correct distortion in image and video

---

## ▶️ How to Run

### 1️⃣ Extract Frames
```bash
python extract_frames.py
###2️⃣ Camera Calibration
```bash
python camera_calibration.py
###3️⃣ Distortion Correction
```bash
python distortion_correction.py
