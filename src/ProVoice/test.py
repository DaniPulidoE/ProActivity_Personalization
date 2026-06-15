import cv2
import numpy as np
import mediapipe as mp
import argparse

mp_face_mesh = mp.solutions.face_mesh

def compute_gaze_score(landmarks, image_width, image_height) -> float:
    # iris landmarks: left eye 468, 469, 470, 471; right eye 473, 474, 475, 476
    left_pts = [landmarks[i] for i in [468, 469, 470, 471]]
    right_pts = [landmarks[i] for i in [473, 474, 475, 476]]
    
    def avg_point(pts):
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        return np.array([np.mean(xs)*image_width, np.mean(ys)*image_height])
    
    left_center = avg_point(left_pts)
    right_center = avg_point(right_pts)
    left_outer = landmarks[33] 
    left_inner = landmarks[133] 
    right_inner = landmarks[362]
    right_outer = landmarks[263]

    left_eye_center = avg_point([left_outer, left_inner])
    right_eye_center = avg_point([right_outer, right_inner])

    left_eye_width = np.linalg.norm((np.array([left_outer.x, left_outer.y]) - 
                                     np.array([left_inner.x, left_inner.y])) * np.array([image_width, image_height]))
    right_eye_width = np.linalg.norm((np.array([right_outer.x, right_outer.y]) - 
                                      np.array([right_inner.x, right_inner.y])) * np.array([image_width, image_height]))

    left_score = np.linalg.norm(left_center - left_eye_center) / left_eye_width
    right_score = np.linalg.norm(right_center - right_eye_center) / right_eye_width

    return float((left_score + right_score) / 2.0)

def gaze_score_on_frame(frame, face_mesh) -> float:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return 0.0
    lm = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    return compute_gaze_score(lm, w, h)

def main():
    parser = argparse.ArgumentParser(description="Gaze Score Test")
    parser.add_argument(
        "--source",
        type=str,
        default="udp",
        choices=["local", "udp"],
        help="Select video source: 'local' (camera) or 'udp' (default)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="udp://127.0.0.1:8554",
        help="Specify the stream address when using UDP"
    )
    args = parser.parse_args()

    if args.source == "local":
        video_source = 0  # local camera ID
        print("Starting: local camera...")
    else:
        video_source = args.url
        print(f"Connecting: UDP stream {video_source} ...")

    cap = cv2.VideoCapture(video_source)

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            score = gaze_score_on_frame(frame, fm)
            print(f"Gaze Score: {score:.3f}")
            cv2.putText(frame, f"Gaze Score: {score:.3f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Gaze Test", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
