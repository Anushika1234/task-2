import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

class RollerCoasterGUI:
    def __init__(self, root, detection_function):
        self.root = root
        self.root.title("Horror Roller Coaster Detection")
        self.root.geometry("800x600")
        self.detection_function = detection_function
        self.photo = None
        self.quit_flag = False

        # Widgets
        self.start_btn = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_btn.pack(pady=10)

        self.stop_btn = tk.Button(root, text="Stop Detection", command=self.stop_detection)
        self.stop_btn.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(root, text="Press 'Start Detection' to begin", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_gui(self, frame, label):
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            image_pil = image_pil.resize((400, 400), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image_pil)
            self.image_label.config(image=self.photo)
        self.result_label.config(text=label)
        self.root.update()

    def start_detection(self):
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.result_label.config(text="Detecting...")
        # Run detection in a separate thread to avoid freezing GUI
        self.root.after(100, lambda: self.detection_function(self.root, self.update_gui))

    def stop_detection(self):
        self.quit_flag = True
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.result_label.config(text="Detection stopped")

    def on_closing(self):
        self.quit_flag = True
        self.root.destroy()
