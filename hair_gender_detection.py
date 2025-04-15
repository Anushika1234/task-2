import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from facenet_pytorch import MTCNN
import torch
import os
from sklearn.model_selection import train_test_split
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

# Step 1: Prepare dataset
data_dir = 'UTKFace'
images = []
ages = []
genders = []

for filename in os.listdir(data_dir):
    if filename.endswith('.jpg'):
        try:
            parts = filename.split('_')
            age = int(parts[0])
            gender = int(parts[1])  # 0: Male, 1: Female
            images.append(os.path.join(data_dir, filename))
            ages.append(age)
            genders.append(gender)
        except:
            continue

df = pd.DataFrame({'image': images, 'age': ages, 'gender': genders})

# Step 2: Preprocess images
def preprocess_image(img_path, size=(128, 128)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

X = np.array([preprocess_image(img) for img in df['image']])
y_age = df['age'].values
y_gender = df['gender'].values

# Step 3: Split data
X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val = train_test_split(
    X, y_age, y_gender, test_size=0.2, random_state=42
)

# Step 4: Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='linear')  # [age, gender_prob]
])

model.compile(optimizer='adam', loss=['mse', 'binary_crossentropy'], loss_weights=[0.5, 0.5])

# Step 5: Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Step 6: Train model
model.fit(
    datagen.flow(X_train, [y_age_train, y_gender_train], batch_size=32),
    validation_data=(X_val, [y_age_val, y_gender_val]),
    epochs=20,
    verbose=1
)

# Step 7: Save model
model.save('age_gender_model.h5')

# Step 8: Hair length detection
def detect_hair_length(image, face_box):
    x1, y1, x2, y2 = map(int, face_box)
    # Convert to grayscale and threshold to detect dark hair
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Analyze region below face (shoulder area)
    shoulder_y = y2 + int((y2 - y1) * 0.5)  # 50% below face
    if shoulder_y >= image.shape[0]:
        return 'Short'
    
    hair_region = thresh[shoulder_y:min(shoulder_y + 50, image.shape[0]), x1:x2]
    hair_pixels = np.sum(hair_region == 255)  # White pixels indicate hair
    total_pixels = hair_region.size
    hair_ratio = hair_pixels / total_pixels
    
    return 'Long' if hair_ratio > 0.3 else 'Short'  # Threshold tuned for hair presence

# Step 9: Predict with logic
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

def predict_with_logic(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return None, None, None, image_rgb
    
    # Process first face
    x1, y1, x2, y2 = map(int, boxes[0])
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None, None, None, image_rgb
    
    # Preprocess for model
    face_processed = cv2.resize(face, (128, 128)) / 255.0
    face_processed = np.expand_dims(face_processed, axis=0)
    
    # Predict age and gender
    age, gender_prob = model.predict(face_processed, verbose=0)[0]
    age = int(age)
    gender = 'Male' if gender_prob > 0.5 else 'Female'
    
    # Detect hair length
    hair_length = detect_hair_length(image, boxes[0])
    
    # Apply logic
    if 20 <= age <= 30:
        gender = 'Female' if hair_length == 'Long' else 'Male'
    
    return age, gender, hair_length, image_rgb

# Step 10: Create GUI
class HairGenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hair-Based Gender Detection")
        self.root.geometry("800x600")
        
        # Widgets
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        
        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)
        
        self.image_path = None
        self.photo = None
        
    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if not self.image_path:
            return
        
        # Predict
        age, gender, hair_length, image_rgb = predict_with_logic(self.image_path)
        
        if age is None:
            messagebox.showerror("Error", "No face detected in the image.")
            return
        
        # Display image
        image_pil = Image.fromarray(image_rgb)
        image_pil = image_pil.resize((400, 400), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=self.photo)
        
        # Display results
        result_text = f"Age: {age}\nGender: {gender}\nHair Length: {hair_length}"
        self.result_label.config(text=result_text)
        
        # Log to CSV
        entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_data = [{
            'Entry_Time': entry_time,
            'Age': age,
            'Gender': gender,
            'Hair_Length': hair_length
        }]
        df_log = pd.DataFrame(log_data)
        output_file = 'hair_gender_log.csv'
        df_log.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

# Step 11: Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = HairGenderApp(root)
    root.mainloop()