import tkinter as tk
from PIL import Image, ImageTk
import cv2

class GUIWindow:
    def __init__(self, width=1280, height=720):
        self.root = tk.Tk()
        self.root.title("Action Recognition to Voice System")
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg='#2b2b2b')
        
        self.is_running = False
        self._create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def _create_widgets(self):
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        video_frame = tk.Frame(main_frame, bg='#1e1e1e', relief=tk.RAISED, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_label = tk.Label(video_frame, text="Webcam Feed", bg='#1e1e1e', fg='white', 
                              font=('Arial', 14, 'bold'))
        video_label.pack(pady=10)
        
        self.video_canvas = tk.Canvas(video_frame, bg='#000000', width=640, height=480)
        self.video_canvas.pack(padx=10, pady=10)
        
        action_frame = tk.Frame(main_frame, bg='#1e1e1e', relief=tk.RAISED, bd=2, width=400)
        action_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        action_frame.pack_propagate(False)
        
        title_label = tk.Label(action_frame, text="Recognized Action", bg='#1e1e1e', 
                              fg='white', font=('Arial', 16, 'bold'))
        title_label.pack(pady=20)
        
        self.action_label = tk.Label(action_frame, text="No action detected", 
                                     bg='#1e1e1e', fg='#4CAF50', 
                                     font=('Arial', 32, 'bold'), 
                                     wraplength=350, justify=tk.CENTER)
        self.action_label.pack(pady=30, padx=20)
        
        self.status_label = tk.Label(action_frame, text="Status: Ready", 
                                     bg='#1e1e1e', fg='#888888', 
                                     font=('Arial', 12))
        self.status_label.pack(pady=10)
        
        instructions_frame = tk.Frame(action_frame, bg='#1e1e1e')
        instructions_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        instructions_text = """
Instructions:
• Position yourself in front of the webcam
• Perform actions clearly
• Actions will be displayed and spoken
• Close window to exit
        """
        instructions_label = tk.Label(instructions_frame, text=instructions_text.strip(),
                                     bg='#1e1e1e', fg='#cccccc', 
                                     font=('Arial', 10),
                                     justify=tk.LEFT, anchor='nw')
        instructions_label.pack(fill=tk.BOTH, expand=True)
        
        actions_list_frame = tk.Frame(action_frame, bg='#1e1e1e')
        actions_list_frame.pack(fill=tk.BOTH, padx=20, pady=10)
        
        actions_title = tk.Label(actions_list_frame, text="Supported Actions:", 
                                bg='#1e1e1e', fg='white', 
                                font=('Arial', 12, 'bold'))
        actions_title.pack(anchor='w')
        
        actions_text = "hello everyone, i want water, i need help, i have headache, i want to go there, i am hungry, good, bad, i have chest pain, i want to sleep"
        actions_label = tk.Label(actions_list_frame, text=actions_text,
                                bg='#1e1e1e', fg='#aaaaaa', 
                                font=('Arial', 9),
                                wraplength=350, justify=tk.LEFT)
        actions_label.pack(anchor='w', pady=5)
        
    def update_frame(self, frame):
        if frame is None:
            return
        
        try:
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 640
                canvas_height = 480
            
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            canvas_aspect = canvas_width / canvas_height
            
            if canvas_aspect > aspect_ratio:
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            else:
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)
            
            self.video_canvas.delete("all")
            self.video_canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
            self.video_canvas.image = photo
        except Exception:
            pass
        
    def update_action(self, action_name, confidence=0.0):
        if action_name:
            self.action_label.config(text=action_name, fg='#4CAF50')
            confidence_text = f"Status: Detected (Confidence: {confidence:.0%})"
            self.status_label.config(text=confidence_text, fg='#4CAF50')
        else:
            self.action_label.config(text="No action detected", fg='#888888')
            self.status_label.config(text="Status: Waiting...", fg='#888888')
    
    def update_status(self, status_text):
        self.status_label.config(text=f"Status: {status_text}")
    
    def run(self):
        self.is_running = True
        self.root.mainloop()
    
    def on_closing(self):
        self.is_running = False
        self.root.quit()
        self.root.destroy()
    
    def is_active(self):
        try:
            return self.root.winfo_exists() and self.is_running
        except:
            return False
