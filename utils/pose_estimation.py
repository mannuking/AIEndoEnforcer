import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

model_path = r"Models/pose_landmarker_lite.task" 

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True 
)
detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def resize_window(frame):
    window_width = 800
    window_height = 600
    original_height, original_width = frame.shape[:2]
    scale_width = window_width / original_width
    scale_height = window_height / original_height
    scale = min(scale_width, scale_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def process_frame_for_pose(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    landmarks = None
    annotated_image = frame

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
        
    resized_image = resize_window(annotated_image)
    return landmarks, resized_image

def get_landmark_coordinates(landmarks):
    if not landmarks:
        return None
    return {
        'left_shoulder': [landmarks[11].x, landmarks[11].y, landmarks[11].z],
        'left_elbow': [landmarks[13].x, landmarks[13].y, landmarks[13].z],
        'left_wrist': [landmarks[15].x, landmarks[15].y, landmarks[15].z],
        'right_shoulder': [landmarks[12].x, landmarks[12].y, landmarks[12].z],
        'right_elbow': [landmarks[14].x, landmarks[14].y, landmarks[14].z],
        'right_wrist': [landmarks[16].x, landmarks[16].y, landmarks[16].z],
        'left_hip': [landmarks[23].x, landmarks[23].y, landmarks[23].z],
        'left_knee': [landmarks[25].x, landmarks[25].y, landmarks[25].z],
        'left_ankle': [landmarks[27].x, landmarks[27].y, landmarks[27].z],
        'right_hip': [landmarks[24].x, landmarks[24].y, landmarks[24].z],
        'right_knee': [landmarks[26].x, landmarks[26].y, landmarks[26].z],
        'right_ankle': [landmarks[28].x, landmarks[28].y, landmarks[28].z],
    }

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def calculate_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

class RepCounter:
    def __init__(self):
        self.squat_count = 0
        self.pushup_count = 0
        self.squat_stage = None
        self.pushup_stage = None
        self.set_count = 0
        self.reps_per_set = 5  # Adjust as needed

    def count_squat(self, angle):
        if angle > 160:
            self.squat_stage = "up"
        elif angle < 100 and self.squat_stage == 'up':
            self.squat_stage = "down"
            self.squat_count += 1
            if self.squat_count % self.reps_per_set == 0:
                self.set_count += 1

    def count_pushup(self, angle):
        if angle > 160:
            self.pushup_stage = "up"
        elif angle < 90 and self.pushup_stage == 'up':
            self.pushup_stage = "down"
            self.pushup_count += 1
            if self.pushup_count % self.reps_per_set == 0:
                self.set_count += 1

def analyze_pose(landmarks, rep_counter):
    coords = get_landmark_coordinates(landmarks)
    if not coords:
        return {}

    # Calculate angles
    left_arm_angle = calculate_angle(coords['left_shoulder'], coords['left_elbow'], coords['left_wrist'])
    right_arm_angle = calculate_angle(coords['right_shoulder'], coords['right_elbow'], coords['right_wrist'])
    left_leg_angle = calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
    right_leg_angle = calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])

    # Calculate distances
    hip_knee_distance = calculate_distance(coords['left_hip'], coords['left_knee'])
    shoulder_wrist_distance = calculate_distance(coords['left_shoulder'], coords['left_wrist'])

    # Count reps
    rep_counter.count_squat(left_leg_angle)
    rep_counter.count_pushup(left_arm_angle)

    return {
        'left_arm_angle': left_arm_angle,
        'right_arm_angle': right_arm_angle,
        'left_leg_angle': left_leg_angle,
        'right_leg_angle': right_leg_angle,
        'hip_knee_distance': hip_knee_distance,
        'shoulder_wrist_distance': shoulder_wrist_distance,
        'squat_count': rep_counter.squat_count,
        'pushup_count': rep_counter.pushup_count,
        'set_count': rep_counter.set_count
    }

def draw_text_on_image(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image

def display_workout_stats(image, stats):
    image = draw_text_on_image(image, f"Squats: {stats['squat_count']}", (10, 30))
    image = draw_text_on_image(image, f"Pushups: {stats['pushup_count']}", (10, 60))
    image = draw_text_on_image(image, f"Sets: {stats['set_count']}", (10, 90))
    return image
