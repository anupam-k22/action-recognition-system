import cv2

class CameraCapture:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
    def initialize(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {self.camera_index}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    def read_frame(self):
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return (ret, cv2.flip(frame, 1)) if ret else (False, None)
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()
