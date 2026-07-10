import cv2
import numpy as np
import sys
import types
import warnings
warnings.filterwarnings('ignore')

class TasksModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.python = types.ModuleType(f'{name}.python')
        sys.modules[f'{name}.python'] = self.python

tasks_module = TasksModule('mediapipe.tasks')
sys.modules['mediapipe.tasks'] = tasks_module
sys.modules['mediapipe.tasks.python'] = tasks_module.python

try:
    import mediapipe as mp
except Exception as e:
    print(f"Warning: MediaPipe import issue: {e}")
    from mediapipe.python.solutions import pose as mp_pose, hands as mp_hands, drawing_utils as mp_drawing
    class MPWrapper:
        class solutions:
            pose, hands, drawing_utils = mp_pose, mp_hands, mp_drawing
    mp = MPWrapper()

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        pose_results = self.pose.process(rgb_frame)
        
        hand_results = self.hands.process(rgb_frame)
        
        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if pose_results.pose_landmarks:
            spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            self.mp_drawing.draw_landmarks(annotated_frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, spec, spec)
        if hand_results.multi_hand_landmarks:
            spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, spec, spec)
        pose_landmarks = pose_results.pose_landmarks
        left_hand_landmarks = right_hand_landmarks = None
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if hand_results.multi_handedness[idx].classification[0].label == "Left":
                    left_hand_landmarks = hand_landmarks
                else:
                    right_hand_landmarks = hand_landmarks
        
        return {
            'pose_landmarks': pose_landmarks,
            'left_hand_landmarks': left_hand_landmarks,
            'right_hand_landmarks': right_hand_landmarks,
            'annotated_frame': annotated_frame
        }
    
    def release(self):
        self.pose.close()
        self.hands.close()
