import time
import traceback
from collections import deque, Counter
from camera_capture import CameraCapture
from pose_detector import PoseDetector
from action_classifier import ActionClassifier
from tts_engine import TTSEngine
from gui_window import GUIWindow

class ActionRecognitionApp:
    def __init__(self):
        self.camera = CameraCapture(camera_index=0, width=640, height=480)
        self.pose_detector = PoseDetector()
        self.action_classifier = ActionClassifier()
        self.tts_engine = TTSEngine(rate=150, volume=0.9)
        self.gui = GUIWindow(width=1280, height=720)
        
        self.is_running = False
        self.last_action = None
        self.action_history = deque(maxlen=5)
        self.action_cooldown = 1.5
        self.last_action_time = 0
        self.stable_action = None
        self.stable_action_start_time = None
        self.stability_duration = 2.0
        self.last_spoken_action = None
        
    def initialize(self):
        print("Initializing Action Recognition System...")
        
        if not self.camera.initialize():
            print("Error: Failed to initialize camera")
            return False
        
        print("Camera initialized successfully")
        print("Starting application...")
        return True
    
    def _process_frame(self):
        if not self.is_running or not self.gui.is_active():
            return
        
        try:
            start_time = time.time()
            
            ret, frame = self.camera.read_frame()
            if not ret:
                print("Failed to read frame from camera")
                self.gui.root.after(100, self._process_frame)
                return
            
            try:
                results = self.pose_detector.process_frame(frame)
                annotated_frame = results['annotated_frame']
            except Exception as e:
                print(f"MediaPipe processing error (showing raw frame): {e}")
                annotated_frame = frame
                results = {'pose_landmarks': None, 'left_hand_landmarks': None, 'right_hand_landmarks': None, 'annotated_frame': frame}
            
            try:
                action_name, confidence = self.action_classifier.classify(
                    results['pose_landmarks'],
                    results['left_hand_landmarks'],
                    results['right_hand_landmarks']
                )
            except Exception as e:
                print(f"Error in action classification: {e}")
                action_name, confidence = None, 0.0
            
            if action_name:
                self.action_history.append(action_name)
                if len(self.action_history) >= 3:
                    most_common = Counter(self.action_history).most_common(1)[0]
                    if most_common[1] >= 2:
                        action_name = most_common[0]
            
            self.gui.update_action(action_name, confidence)
            
            current_time = time.time()
            
            if action_name:
                if action_name != self.stable_action:
                    self.stable_action = action_name
                    self.stable_action_start_time = current_time
                    print(f"Action detected (waiting for stability): {action_name} (confidence: {confidence:.2f})")
                
                elif self.stable_action_start_time is not None:
                    elapsed_stable_time = current_time - self.stable_action_start_time
                    
                    if elapsed_stable_time >= self.stability_duration:
                        if action_name != self.last_spoken_action:
                            if (current_time - self.last_action_time) >= self.action_cooldown:
                                self.tts_engine.speak_action(action_name)
                                self.last_spoken_action = action_name
                                self.last_action_time = current_time
                                print(f"Action confirmed and spoken: {action_name} (confidence: {confidence:.2f})")
            else:
                if self.stable_action is not None:
                    self.stable_action = None
                    self.stable_action_start_time = None
            
            self.last_action = action_name
            
            self.gui.update_frame(annotated_frame)
            
            elapsed_time = time.time() - start_time
            delay = max(1, int((0.033 - elapsed_time) * 1000))
            self.gui.root.after(delay, self._process_frame)
            
        except Exception as e:
            print(f"Error in frame processing: {e}")
            traceback.print_exc()
            self.gui.root.after(50, self._process_frame)
    
    def run(self):
        if not self.initialize():
            return
        
        self.is_running = True
        self.gui.update_status("Initializing...")
        
        print("Application started. Close the window to exit.")
        self.gui.root.after(100, self._process_frame)
        self.gui.update_status("Running")
        
        try:
            self.gui.run()
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("Cleaning up resources...")
        self.is_running = False
        self.camera.release()
        self.pose_detector.release()
        self.tts_engine.stop()
        if self.gui.is_active():
            self.gui.on_closing()
        print("Cleanup complete.")


def main():
    app = ActionRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()
