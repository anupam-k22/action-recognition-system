import threading
import queue
import time
import pyttsx3

class TTSEngine:
    def __init__(self, rate=150, volume=0.9):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
        except Exception as e:
            print(f"Warning: TTS engine initialization failed: {e}")
            print("Voice output will be disabled.")
            self.engine = None
        
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = None
        self.stop_flag = False
        self.last_action = None
        self.cooldown_time = 2.0
        self.last_action_time = 0
        
    def speak(self, text):
        if self.engine is None or not (text and text.strip()):
            return
        self.speech_queue.put(text)
        if not self.is_speaking:
            self._start_speech_thread()
    
    def _start_speech_thread(self):
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
            self.speech_thread.start()
    
    def _process_speech_queue(self):
        self.is_speaking = True
        while not self.stop_flag:
            try:
                text = self.speech_queue.get(timeout=0.5)
                current_time = time.time()
                if text == self.last_action and (current_time - self.last_action_time) < self.cooldown_time:
                    self.speech_queue.task_done()
                    continue
                if self.engine is not None:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.last_action, self.last_action_time = text, current_time
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in TTS: {e}")
                continue
        
        self.is_speaking = False
    
    def speak_action(self, action_name):
        if action_name:
            self.speak(action_name)

    def stop(self):
        self.stop_flag = True
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)
        
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass
