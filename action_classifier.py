import numpy as np

class ActionClassifier:
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    WRIST = 0
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_normalized_landmarks(landmarks):
        if landmarks is None:
            return None
        
        return np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark]
        )
    
    @staticmethod
    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
        
    def classify(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks):
        if pose_landmarks is None:
            return None, 0.0
        
        pose_norm = self.get_normalized_landmarks(pose_landmarks)
        left_hand_norm = self.get_normalized_landmarks(left_hand_landmarks) if left_hand_landmarks else None
        right_hand_norm = self.get_normalized_landmarks(right_hand_landmarks) if right_hand_landmarks else None
        
        try:
            nose = pose_norm[self.NOSE]
            left_eye = pose_norm[self.LEFT_EYE]
            right_eye = pose_norm[self.RIGHT_EYE]
            left_shoulder = pose_norm[self.LEFT_SHOULDER]
            right_shoulder = pose_norm[self.RIGHT_SHOULDER]
            left_wrist = pose_norm[self.LEFT_WRIST]
            right_wrist = pose_norm[self.RIGHT_WRIST]
        except (IndexError, KeyError):
            return None, 0.0
        
        left_wrist, right_wrist, left_hand_norm, right_hand_norm = self._correct_hand_detection(
            left_wrist, right_wrist, left_hand_norm, right_hand_norm, left_shoulder, right_shoulder
        )
        
        action_scores = {
            'hello everyone': self._check_hello_everyone(left_wrist, right_wrist, left_shoulder, right_shoulder),
            'i want water': self._check_i_want_water(left_wrist, right_wrist, nose, left_hand_norm, right_hand_norm),
            'i need help': self._check_i_need_help(left_wrist, right_wrist, left_shoulder, right_shoulder),
            'i have headache': self._check_headache(right_wrist, nose),
            'i want to go there': self._check_i_want_to_go_there(left_wrist, left_hand_norm, left_shoulder),
            'i am hungry': self._check_i_am_hungry(right_wrist, left_shoulder, nose),
            'good': self._check_good(left_hand_norm, right_hand_norm),
            'bad': self._check_bad(left_hand_norm, right_hand_norm),
            'i have chest pain': self._check_i_have_chest_pain(right_wrist, left_shoulder, nose),
            'i want to sleep': self._check_i_want_to_sleep(left_wrist, right_wrist, left_shoulder, nose, left_eye, right_eye)
        }
        
        best_action = max(action_scores, key=action_scores.get)
        best_score = action_scores[best_action]
        
        if best_score > 0.5:
            return best_action, best_score
        return None, 0.0
    
    def _correct_hand_detection(self, left_wrist, right_wrist, left_hand_norm, right_hand_norm, left_shoulder, right_shoulder):
        if left_wrist[0] > 0 and right_wrist[0] > 0:
            left_dist_to_left_shoulder = abs(left_wrist[0] - left_shoulder[0])
            left_dist_to_right_shoulder = abs(left_wrist[0] - right_shoulder[0])
            
            right_dist_to_right_shoulder = abs(right_wrist[0] - right_shoulder[0])
            right_dist_to_left_shoulder = abs(right_wrist[0] - left_shoulder[0])
            
            if (left_dist_to_right_shoulder < left_dist_to_left_shoulder and 
                right_dist_to_left_shoulder < right_dist_to_right_shoulder):
                
                print("Hand detection corrected: swapping left/right hands")
                return right_wrist, left_wrist, right_hand_norm, left_hand_norm
        
        elif left_wrist[0] > 0 and right_wrist[0] <= 0:
            if left_wrist[0] > right_shoulder[0]:
                print("Hand detection corrected: moving left hand to right")
                return right_wrist, left_wrist, right_hand_norm, left_hand_norm
                
        elif right_wrist[0] > 0 and left_wrist[0] <= 0:
            if right_wrist[0] < left_shoulder[0]:
                print("Hand detection corrected: moving right hand to left")
                return right_wrist, left_wrist, right_hand_norm, left_hand_norm
        
        return left_wrist, right_wrist, left_hand_norm, right_hand_norm
    
    def _check_hello_everyone(self, left_wrist, right_wrist, left_shoulder, right_shoulder):
        shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        right_raised = right_wrist[1] < shoulder_avg_y - 0.1 if right_wrist[0] > 0 else False
        left_not_raised = left_wrist[1] > shoulder_avg_y if left_wrist[0] > 0 else True
        
        if right_raised and left_not_raised:
            return 0.8
        return 0.0
    
    def _check_i_want_water(self, left_wrist, right_wrist, nose, left_hand, right_hand):
        mouth_y, mouth_x = nose[1] + 0.03, nose[0]
        for wrist in (left_wrist, right_wrist):
            if wrist[0] > 0 and abs(wrist[1] - mouth_y) < 0.1 and abs(wrist[0] - mouth_x) < 0.18:
                if wrist[2] < nose[2] + 0.15 and wrist[1] <= mouth_y + 0.08:
                    return 0.9
        return 0.0
    
    def _check_i_need_help(self, left_wrist, right_wrist, left_shoulder, right_shoulder):
        shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        left_raised = left_wrist[1] < shoulder_avg_y - 0.2 if left_wrist[0] > 0 else False
        right_raised = right_wrist[1] < shoulder_avg_y - 0.2 if right_wrist[0] > 0 else False
        
        if left_raised and right_raised:
            return 0.85
        return 0.0
    
    def _check_headache(self, right_wrist, nose):
        forehead_y = nose[1] - 0.05
        threshold = 0.08
        
        if right_wrist[0] > 0:
            head_dist = abs(right_wrist[1] - forehead_y)
            if head_dist < threshold and right_wrist[2] < nose[2] + 0.1:
                return 0.85
        return 0.0
    
    def _check_i_want_to_go_there(self, left_wrist, left_hand, left_shoulder):
        if left_hand is None or left_wrist[0] <= 0:
            return 0.0
        return 0.85 if left_hand[self.INDEX_TIP][1] < left_hand[self.WRIST][1] - 0.03 and left_wrist[1] < left_shoulder[1] + 0.3 else 0.0
    
    def _check_i_am_hungry(self, wrist, left_shoulder, nose):
        if wrist[0] > 0:
            stomach_x, stomach_y = nose[0], left_shoulder[1] + 0.25
            if abs(wrist[0] - stomach_x) < 0.15 and abs(wrist[1] - stomach_y) < 0.15 and wrist[2] < nose[2] + 0.2:
                return 0.85
        return 0.0
    
    def _check_good(self, left_hand, right_hand):
        for hand in (left_hand, right_hand):
            if hand is None:
                continue
            
            thumb_tip = hand[self.THUMB_TIP]
            thumb_ip = hand[self.THUMB_IP]
            wrist = hand[self.WRIST]
            index_tip = hand[self.INDEX_TIP]
            index_pip = hand[self.INDEX_PIP]
            middle_tip = hand[self.MIDDLE_TIP]
            middle_pip = hand[self.MIDDLE_PIP]
            
            thumb_up = thumb_tip[1] < wrist[1] - 0.03
            thumb_extended = self.calculate_distance(thumb_tip[:2], thumb_ip[:2]) > 0.04
            index_closed = self.calculate_distance(index_tip[:2], index_pip[:2]) < 0.035
            middle_closed = self.calculate_distance(middle_tip[:2], middle_pip[:2]) < 0.035
            
            if thumb_up and thumb_extended and index_closed and middle_closed:
                return 0.9
        
        return 0.0
    
    def _check_bad(self, left_hand, right_hand):
        for hand in (left_hand, right_hand):
            if hand is None:
                continue
            
            thumb_tip = hand[self.THUMB_TIP]
            thumb_ip = hand[self.THUMB_IP]
            wrist = hand[self.WRIST]
            index_tip = hand[self.INDEX_TIP]
            index_pip = hand[self.INDEX_PIP]
            middle_tip = hand[self.MIDDLE_TIP]
            middle_pip = hand[self.MIDDLE_PIP]
            
            thumb_down = thumb_tip[1] > wrist[1] + 0.02
            thumb_extended = self.calculate_distance(thumb_tip[:2], thumb_ip[:2]) > 0.025
            index_closed = self.calculate_distance(index_tip[:2], index_pip[:2]) < 0.04
            middle_closed = self.calculate_distance(middle_tip[:2], middle_pip[:2]) < 0.04
            
            if thumb_down and thumb_extended and (index_closed or middle_closed):
                return 0.9
        
        return 0.0
    
    def _check_i_have_chest_pain(self, wrist, left_shoulder, nose):
        if wrist[0] > 0:
            chest_x, chest_y = nose[0], left_shoulder[1] + 0.12
            if abs(wrist[0] - chest_x) < 0.10 and abs(wrist[1] - chest_y) < 0.10 and wrist[2] < nose[2] + 0.2:
                return 0.85
        return 0.0
    
    def _check_i_want_to_sleep(self, left_wrist, right_wrist, left_shoulder, nose, left_eye, right_eye):
        cheek_x, cheek_y = nose[0] - 0.06, nose[1] + 0.02
        for wrist in (left_wrist, right_wrist):
            if wrist[0] > 0:
                if self.calculate_distance(wrist[:2], [cheek_x, cheek_y]) < 0.20 and wrist[2] < nose[2] + 0.3 and wrist[1] > nose[1] - 0.02:
                    return 0.8
        return 0.0
