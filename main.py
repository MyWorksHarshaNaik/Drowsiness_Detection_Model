import tkinter as tk
from tkinter import filedialog, Label, Button
import tkinter.messagebox as messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import threading
from tensorflow.keras.models import load_model

# Load pre-trained models
face_cascade = cv2.CascadeClassifier('./Utils/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./Utils/haarcascade_eye.xml')
age_model = cv2.dnn.readNet('./Utils/age_net.caffemodel', './Utils/age_deploy.prototxt')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load the trained model for eyes open/closed detection
model = load_model('./Models/model1.h5')

# Dictionary to track the time eyes have been closed for each face
closed_eye_start_time = {}
LOCK = threading.Lock()

class DrowsinessDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection System")

        self.label = Label(root, text="Choose an image or video file to analyze")
        self.label.pack(pady=10)

        self.img_button = Button(root, text="Select Image", command=self.predict_image)
        self.img_button.pack(pady=5)

        self.vid_button = Button(root, text="Select Video", command=self.predict_video)
        self.vid_button.pack(pady=5)

        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()

    def process_frame(self, frame):
        """Process a single frame to detect drowsiness and age."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        current_time = time.time()

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
            closed_eyes_count = 0

            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_color[ey:ey + eh, ex:ex + ew]
                eye_roi_resized = cv2.resize(eye_roi, (86, 86))  # Match input size of the model
                eye_roi_normalized = eye_roi_resized / 255.0
                eye_roi_expanded = np.expand_dims(eye_roi_normalized, axis=0)

                prediction = model.predict(eye_roi_expanded)
                if prediction[0][0] < 0.5:  # Assuming <0.5 indicates closed eyes
                    closed_eyes_count += 1

            face_id = (x, y, w, h)

            with LOCK:
                if closed_eyes_count >= len(eyes):
                    if face_id not in closed_eye_start_time:
                        closed_eye_start_time[face_id] = current_time

                    elapsed_time = current_time - closed_eye_start_time[face_id]

                    # Mark with a red bounding box if eyes closed for more than 2 seconds
                    if elapsed_time > 2:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        label = "Drowsy"

                        # Show pop-up alert
                        if elapsed_time < 2.5:  # Show the alert only once when the threshold is crossed
                            messagebox.showwarning("Drowsiness Alert", "Eyes closed for more than 2 seconds!")
                    else:
                        label = "Blinking"
                else:
                    closed_eye_start_time.pop(face_id, None)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = "Alert"

                # Age detection
                blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                age_model.setInput(blob)
                age_predictions = age_model.forward()
                age = age_list[np.argmax(age_predictions[0])]

                # Annotate the frame
                cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {label}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def predict_image(self):
        """Process an image file for drowsiness detection."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            frame = cv2.imread(file_path)
            processed_frame = self.process_frame(frame)

            # Convert processed frame to display in Tkinter
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((800, 600))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk

    def predict_video(self):
        """Process a video file for drowsiness detection."""
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            cap = cv2.VideoCapture(file_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)

                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((800, 600))
                img_tk = ImageTk.PhotoImage(img)

                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk
                self.root.update()

            cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessDetectionApp(root)
    root.mainloop()
