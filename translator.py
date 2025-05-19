import os
import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, Frame, Button, Label, Entry, Text, messagebox, StringVar, filedialog, ttk
import time
import PIL.Image
import PIL.ImageTk
import threading
import logging
import pickle
import uuid

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RealtimeTranslator:
    def __init__(self, parent):
        self.frame = Frame(parent, bg="#D3D3D3")
        self.frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.model_mixed = None
        self.labels_dict_mixed = None
        self.label_mapping = {}
        self.inverse_label_mapping = {}
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.text_output = ""
        self.last_update_time = time.time()
        self.running = True
        self.frame_data = None
        self.lock = threading.Lock()
        self.is_waiting = False
        self.wait_start_time = None
        self.wait_duration = 3
        self.recording = False
        self.camera_options = []
        self.selected_camera = StringVar()
        self.ip_camera_url = ""  # IP kamera URL manzilini saqlash uchun o‘zgaruvchi
        self.detect_cameras()
        self.setup_gui()

    def detect_cameras(self):
        self.camera_options = ["IP Camera"]
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            self.camera_options.append(f"Camera {index}")
            cap.release()
            index += 1

    def setup_gui(self):
        # Main layout: Left for controls, Right for video
        self.main_frame = Frame(self.frame, bg="#D3D3D3")
        self.main_frame.pack(fill="both", expand=True)

        # Left side: Controls
        self.control_frame = Frame(self.main_frame, bg="#D3D3D3")
        self.control_frame.grid(row=0, column=0, sticky="nsw", padx=10, pady=10)

        # Right side: Video
        self.video_frame = Frame(self.main_frame, bg="#D3D3D3", bd=2, relief="groove")
        self.video_frame.grid(row=0, column=1, sticky="ne", padx=10, pady=10)
        self.video_label = Label(self.video_frame, bg="#D3D3D3")
        self.video_label.pack()

        # Header in control frame
        Label(self.control_frame, text="Realtime Translator", font=("Arial", 16, "bold"), bg="#D3D3D3").pack(pady=10)

        # Camera Selection Section
        camera_frame = Frame(self.control_frame, bg="#D3D3D3", bd=2, relief="groove")
        camera_frame.pack(fill="x", pady=5)
        Label(camera_frame, text="Select Camera:", font=("Arial", 12), bg="#D3D3D3").grid(row=0, column=0, padx=5, pady=5)
        self.camera_menu = ttk.OptionMenu(camera_frame, self.selected_camera, self.camera_options[0] if self.camera_options else "No Camera", *self.camera_options, command=self.select_camera)
        self.camera_menu.grid(row=0, column=1, padx=5, pady=5)
        Button(camera_frame, text="Refresh Camera", command=self.refresh_camera, bg="#FFFFFF", font=("Arial", 10), relief="flat", borderwidth=2).grid(row=0, column=2, padx=5, pady=5)
        self.ip_entry = Entry(camera_frame, font=("Arial", 12))
        self.ip_entry.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        self.ip_entry.insert(0, "Enter IP Camera URL")
        self.ip_entry.grid_remove()  # Hide initially
        self.selected_camera.trace("w", self.toggle_ip_entry)

        # Model Selection Section
        model_frame = Frame(self.control_frame, bg="#D3D3D3", bd=2, relief="groove")
        model_frame.pack(fill="x", pady=5)
        Button(model_frame, text="Select Model File", command=self.select_model_file, bg="#FFFFFF", font=("Arial", 12), relief="flat", borderwidth=2).grid(row=0, column=0, padx=5, pady=5)
        self.model_label = Label(model_frame, text="No model selected", font=("Arial", 12), bg="#D3D3D3")
        self.model_label.grid(row=0, column=1, padx=5, pady=5)

        # Control Sections
        control_inner_frame = Frame(self.control_frame, bg="#D3D3D3")
        control_inner_frame.pack(fill="x", pady=5)

        # Realtime Translation Section
        realtime_frame = Frame(control_inner_frame, bg="#D3D3D3", bd=2, relief="groove")
        realtime_frame.pack(fill="x", pady=5)
        Label(realtime_frame, text="Realtime Translation", font=("Arial", 12, "bold"), bg="#D3D3D3").grid(row=0, column=0, columnspan=2, pady=5)
        Label(realtime_frame, text="Update Interval (seconds):", font=("Arial", 10), bg="#D3D3D3").grid(row=1, column=0, padx=5, pady=5)
        self.interval_entry = Entry(realtime_frame, font=("Arial", 10), width=10)
        self.interval_entry.insert(0, "3")
        self.interval_entry.grid(row=1, column=1, padx=5, pady=5)
        Button(realtime_frame, text="Start", command=self.start_realtime_translation, bg="#FFFFFF", font=("Arial", 10), relief="flat", borderwidth=2).grid(row=2, column=0, columnspan=2, pady=5)

        # Recording Section
        record_frame = Frame(control_inner_frame, bg="#D3D3D3", bd=2, relief="groove")
        record_frame.pack(fill="x", pady=5)
        Label(record_frame, text="Record & Translate", font=("Arial", 12, "bold"), bg="#D3D3D3").grid(row=0, column=0, columnspan=2, pady=5)
        Button(record_frame, text="Start Recording", command=self.start_recording, bg="#FFFFFF", font=("Arial", 10), relief="flat", borderwidth=2).grid(row=1, column=0, padx=5, pady=5)
        Button(record_frame, text="Stop Recording", command=self.stop_recording, bg="#FFFFFF", font=("Arial", 10), relief="flat", borderwidth=2).grid(row=1, column=1, padx=5, pady=5)

        # Other Functions Section
        other_frame = Frame(control_inner_frame, bg="#D3D3D3", bd=2, relief="groove")
        other_frame.pack(fill="x", pady=5)
        Label(other_frame, text="Other Functions", font=("Arial", 12, "bold"), bg="#D3D3D3").grid(row=0, column=0, columnspan=2, pady=5)
        Button(other_frame, text="Translate from Video File", command=self.translate_from_video_file, bg="#FFFFFF", font=("Arial", 10), relief="flat", borderwidth=2).grid(row=1, column=0, columnspan=2, pady=5)
        Button(other_frame, text="Test Mode", command=self.test_mode, bg="#FFFFFF", font=("Arial", 10), relief="flat", borderwidth=2).grid(row=2, column=0, columnspan=2, pady=5)
        Button(other_frame, text="Test Movement Detection", command=self.test_movement_detection, bg="#FFFFFF", font=("Arial", 10), relief="flat", borderwidth=2).grid(row=3, column=0, columnspan=2, pady=5)
        Button(other_frame, text="Stop", command=self.stop_translation, bg="#FF4040", font=("Arial", 10), relief="flat", borderwidth=2).grid(row=4, column=0, columnspan=2, pady=5)

        # Output Section
        output_frame = Frame(self.control_frame, bg="#D3D3D3", bd=2, relief="groove")
        output_frame.pack(fill="x", pady=5)
        Label(output_frame, text="Output:", font=("Arial", 12), bg="#D3D3D3").grid(row=0, column=0, padx=5, pady=5)
        self.output_text = Text(output_frame, height=5, width=50, font=("Arial", 10))
        self.output_text.grid(row=1, column=0, padx=5, pady=5)
        Button(output_frame, text="Clear", command=self.clear_output, bg="#FFFFFF", font=("Arial", 10), relief="flat", borderwidth=2).grid(row=1, column=1, padx=5, pady=5)

    def toggle_ip_entry(self, *args):
        if self.selected_camera.get() == "IP Camera":
            self.ip_entry.grid()
        else:
            self.ip_entry.grid_remove()

    def select_camera(self, *args):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.selected_camera.get() == "IP Camera":
            url = self.ip_entry.get()
            if url == "Enter IP Camera URL" or not url:
                messagebox.showerror("Error", "Please enter a valid IP Camera URL!")
                return
            self.ip_camera_url = url  # URL manzilini saqlaymiz
            self.cap = cv2.VideoCapture(url)
        else:
            try:
                camera_index = int(self.selected_camera.get().split()[-1]) if "Camera" in self.selected_camera.get() else 0
                self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)  # V4L2 backenddan foydalanish
            except Exception as e:
                messagebox.showerror("Error", f"Failed to select camera: {str(e)}")
                self.cap = None
                return
        # Kamera sozlamalari
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Kamerani bir necha marta sinab ko‘rish
        for _ in range(5):  # 5 marta urinib ko‘ramiz
            if self.cap.isOpened():
                break
            time.sleep(0.1)  # Kamera ochilishi uchun biroz kutamiz
        else:
            messagebox.showerror("Error", f"Camera {self.selected_camera.get()} could not be opened!")
            self.cap.release()
            self.cap = None
            return
        if not hasattr(self, 'video_thread'):
            self.start_video_thread()
        self.update_frame()

    def refresh_camera(self):
        if self.selected_camera.get() == "No Camera" or not self.selected_camera.get():
            messagebox.showinfo("Info", "No camera is currently selected. Please select a camera first.")
            return
        # Joriy video oqimini to‘xtatamiz
        self.running = False
        if hasattr(self, 'video_thread'):
            self.video_thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame_data = None
        time.sleep(0.5)  # Tizimga resurslarni bo‘shatish uchun vaqt beramiz
        # Agar IP kamera tanlangan bo‘lsa, URL manzilidan foydalanamiz
        if self.selected_camera.get() == "IP Camera":
            if not self.ip_camera_url:
                messagebox.showerror("Error", "No IP Camera URL provided. Please select the camera again.")
                return
            self.cap = cv2.VideoCapture(self.ip_camera_url)
        else:
            # Lokal kamera uchun qayta aniqlaymiz
            self.detect_cameras()
            self.camera_menu['menu'].delete(0, 'end')
            for option in self.camera_options:
                self.camera_menu['menu'].add_command(label=option, command=lambda value=option: self.selected_camera.set(value))
            self.selected_camera.set(self.camera_options[0] if self.camera_options else "No Camera")
            try:
                camera_index = int(self.selected_camera.get().split()[-1]) if "Camera" in self.selected_camera.get() else 0
                self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to refresh camera: {str(e)}")
                return
        # Kamera sozlamalari
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(5):
                if self.cap.isOpened():
                    break
                time.sleep(0.1)
            else:
                messagebox.showerror("Error", "Failed to refresh camera. Please check if the camera is available.")
                self.cap.release()
                self.cap = None
                return
        self.running = True
        if not hasattr(self, 'video_thread'):
            self.start_video_thread()
        self.update_frame()
        if self.cap is not None and self.cap.isOpened():
            messagebox.showinfo("Success", "Camera refreshed successfully!")
        else:
            messagebox.showerror("Error", "Failed to refresh camera. Please check if the camera is available.")

    def select_model_file(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Pickle files", "*.p")]
        )
        if model_path:
            if self.load_model(model_path):
                self.model_label.config(text=f"Selected: {os.path.basename(model_path)}")
            else:
                self.model_label.config(text="Failed to load model")

    def load_model(self, model_path):
        try:
            model_data = pickle.load(open(model_path, 'rb'))
            self.model_mixed = model_data['model']
            self.label_mapping = model_data.get('label_mapping', {})
            self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            return False

    def clear_output(self):
        self.text_output = ""
        self.output_text.delete(1.0, "end")
        self.output_text.insert(1.0, "Translating: ")

    def start_video_thread(self):
        self.video_thread = threading.Thread(target=self.capture_video, daemon=True)
        self.video_thread.start()

    def capture_video(self):
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Camera failed to capture frames!")
                time.sleep(0.1)  # Kameradan kadr olinmasa biroz kutamiz
                continue
            with self.lock:
                self.frame_data = frame.copy()
            time.sleep(0.01)

    def extract_landmarks(self, frame, draw_skeleton=True):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.5, beta=15)

            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)
            landmarks_list = []
            frame_with_skeleton = frame.copy()
            bounding_boxes = []
            hands_detected = False

            data_aux = []
            if hand_results.multi_hand_landmarks:
                hands_detected = True
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    if draw_skeleton:
                        self.mp_drawing.draw_landmarks(
                            frame_with_skeleton,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                    x_min = int(min(x_coords) * w) - 20
                    x_max = int(max(x_coords) * w) + 20
                    y_min = int(min(y_coords) * h) - 20
                    y_max = int(max(y_coords) * h) + 20
                    bounding_boxes.append((x_min, y_min, x_max, y_max))
                    if draw_skeleton:
                        cv2.rectangle(frame_with_skeleton, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    x_ = [landmark.x for landmark in hand_landmarks.landmark]
                    y_ = [landmark.y for landmark in hand_landmarks.landmark]
                    if x_ and y_:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

            if pose_results.pose_landmarks:
                pose_landmarks = pose_results.pose_landmarks.landmark
                left_elbow = pose_landmarks[13]
                data_aux.append(left_elbow.x * w)
                data_aux.append(left_elbow.y * h)
                right_elbow = pose_landmarks[14]
                data_aux.append(right_elbow.x * w)
                data_aux.append(right_elbow.y * h)
                if draw_skeleton:
                    self.mp_drawing.draw_landmarks(
                        frame_with_skeleton,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )

            if len(data_aux) < 88:
                data_aux.extend([0.0] * (88 - len(data_aux)))

            landmarks_list.append(data_aux)
            logging.debug(f"Extracted landmarks: {len(data_aux)}")
            return landmarks_list, frame_with_skeleton, bounding_boxes, hands_detected
        except Exception as e:
            logging.error(f"Error extracting landmarks: {e}")
            return [], frame, [], False

    def update_frame(self):
        try:
            with self.lock:
                if self.frame_data is not None:
                    frame = self.frame_data.copy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.current_image = PIL.Image.fromarray(frame)
                    self.photo = PIL.ImageTk.PhotoImage(image=self.current_image)
                    self.video_label.config(image=self.photo)
                    self.video_label.image = self.photo
        except Exception as e:
            logging.error(f"Error updating frame: {e}")
        finally:
            self.frame.after(50, self.update_frame)

    def start_realtime_translation(self):
        if self.model_mixed is None:
            messagebox.showerror("Error", "Please select a model file first!")
            return
        try:
            self.interval = float(self.interval_entry.get())
            if self.interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a positive time interval!")
            return
        self.last_update_time = time.time()
        self.running = True
        self.is_waiting = False
        self.wait_start_time = None
        self.translate_loop()

    def stop_translation(self):
        self.running = False
        self.recording = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame_data = None
        self.output_text.delete(1.0, "end")
        self.output_text.insert(1.0, "Translating: (Stopped)")
        self.text_output = ""
        # Restart camera after stopping
        if self.selected_camera.get():
            self.select_camera()

    def start_recording(self):
        if self.model_mixed is None:
            messagebox.showerror("Error", "Please select a model file first!")
            return
        if self.recording:
            messagebox.showinfo("Info", "Recording already in progress!")
            return
        self.recording = True
        self.running = True
        self.output_text.delete(1.0, "end")
        self.output_text.insert(1.0, "Recording: Press 'Stop Recording' to stop (Max 2 minutes)")
        threading.Thread(target=self.record_and_translate, daemon=True).start()

    def stop_recording(self):
        self.recording = False

    def record_and_translate(self):
        temp_video = f"temp_{uuid.uuid4().hex}.avi"
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
        start_time = time.time()
        max_duration = 120  # 2 daqiqa (120 soniya)
        landmarks_history = []
        motion_detected = False
        expected_length = None
        min_frames = 30

        while self.recording and (time.time() - start_time) < max_duration:
            with self.lock:
                if self.frame_data is None:
                    continue
                frame = self.frame_data.copy()
            out.write(frame)
            landmarks, _, _, hands_detected = self.extract_landmarks(frame, draw_skeleton=False)
            if hands_detected and landmarks:
                if expected_length is None:
                    expected_length = len(landmarks[0])
                if len(landmarks[0]) == expected_length:
                    if not motion_detected and len(landmarks_history) > 1:
                        prev_data = np.array(landmarks_history[-1])
                        curr_data = np.array(landmarks[0])
                        if len(curr_data) == len(prev_data) and np.linalg.norm(curr_data - prev_data) > 0.01:
                            motion_detected = True
                    landmarks_history.append(landmarks[0])
            time.sleep(0.01)

        out.release()
        self.recording = False
        self.process_recorded_video(temp_video, landmarks_history)
        os.remove(temp_video)

    def translate_from_video_file(self):
        if self.model_mixed is None:
            messagebox.showerror("Error", "Please select a model file first!")
            return
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not video_path:
            messagebox.showerror("Error", "No video file selected!")
            return

        self.output_text.delete(1.0, "end")
        self.output_text.insert(1.0, "Translating from video file...")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, 60)
        landmarks_history = []
        motion_detected = False
        start_frame = 0
        end_frame = None
        expected_length = None
        min_frames = 30
        prev_landmarks = None
        smoothing_factor = 0.7

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks, _, _, hands_detected = self.extract_landmarks(frame, draw_skeleton=False)
            if hands_detected and landmarks:
                if expected_length is None:
                    expected_length = len(landmarks[0])
                if len(landmarks[0]) == expected_length:
                    if prev_landmarks is not None:
                        smoothed_landmarks = smoothing_factor * prev_landmarks + (1 - smoothing_factor) * np.array(landmarks[0])
                        landmarks[0] = smoothed_landmarks.tolist()
                    if not motion_detected and len(landmarks_history) > 1:
                        prev_data = np.array(landmarks_history[-1])
                        curr_data = np.array(landmarks[0])
                        if len(curr_data) == len(prev_data) and np.linalg.norm(curr_data - prev_data) > 0.01:
                            motion_detected = True
                            start_frame = len(landmarks_history)
                    landmarks_history.append(landmarks[0])
                    if motion_detected and len(landmarks_history) > 1 and len(landmarks_history) >= min_frames:
                        curr_data = np.array(landmarks[0])
                        prev_data = np.array(landmarks_history[-2])
                        if len(curr_data) == len(prev_data) and np.linalg.norm(curr_data - prev_data) < 0.002:
                            end_frame = len(landmarks_history) - 1
                            break
                    prev_landmarks = np.array(landmarks[0])

        cap.release()

        if start_frame < len(landmarks_history) and (end_frame is None or start_frame < end_frame):
            if end_frame is None:
                end_frame = len(landmarks_history) - 1
            frame_features = landmarks_history[start_frame:end_frame + 1]
            expected_length = len(frame_features[0])
            frame_features = [f for f in frame_features if len(f) == expected_length]
            if not frame_features:
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, "Translating: No consistent data detected")
                return
            avg_features = np.mean(frame_features, axis=0)
            if self.model_mixed and len(avg_features) == 88:
                prediction = self.model_mixed.predict([avg_features])
                predicted_idx = prediction[0]
                predicted_word = self.inverse_label_mapping.get(predicted_idx, "Unknown")
            else:
                predicted_word = "Model not loaded or incorrect feature length"
            self.text_output += predicted_word + " "
            self.output_text.delete(1.0, "end")
            self.output_text.insert(1.0, f"Translating: {self.text_output}")
        else:
            self.output_text.delete(1.0, "end")
            self.output_text.insert(1.0, "Translating: No valid data detected")

    def process_recorded_video(self, video_path, live_landmarks):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, 60)
        frame_features = []
        landmarks_history = live_landmarks if live_landmarks else []
        motion_detected = len(landmarks_history) > 1
        start_frame = 0
        end_frame = len(landmarks_history) - 1
        expected_length = None
        min_frames = 30
        prev_landmarks = None
        smoothing_factor = 0.7

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks, _, _, hands_detected = self.extract_landmarks(frame, draw_skeleton=False)
            if hands_detected and landmarks:
                if expected_length is None:
                    expected_length = len(landmarks[0])
                if len(landmarks[0]) == expected_length:
                    if prev_landmarks is not None:
                        smoothed_landmarks = smoothing_factor * prev_landmarks + (1 - smoothing_factor) * np.array(landmarks[0])
                        landmarks[0] = smoothed_landmarks.tolist()
                    if not motion_detected and len(landmarks_history) > 1:
                        prev_data = np.array(landmarks_history[-1])
                        curr_data = np.array(landmarks[0])
                        if len(curr_data) == len(prev_data) and np.linalg.norm(curr_data - prev_data) > 0.01:
                            motion_detected = True
                            start_frame = len(landmarks_history)
                    landmarks_history.append(landmarks[0])
                    if motion_detected and len(landmarks_history) > 1 and len(landmarks_history) >= min_frames:
                        curr_data = np.array(landmarks[0])
                        prev_data = np.array(landmarks_history[-2])
                        if len(curr_data) == len(prev_data) and np.linalg.norm(curr_data - prev_data) < 0.002:
                            end_frame = len(landmarks_history) - 1
                            break
                    prev_landmarks = np.array(landmarks[0])

        cap.release()
        if start_frame < end_frame and len(landmarks_history[start_frame:end_frame + 1]) > 0:
            frame_features = landmarks_history[start_frame:end_frame + 1]
            expected_length = len(frame_features[0])
            frame_features = [f for f in frame_features if len(f) == expected_length]
            if not frame_features:
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, "Translating: No consistent data detected")
                return
            avg_features = np.mean(frame_features, axis=0)
            if self.model_mixed and len(avg_features) == 88:
                prediction = self.model_mixed.predict([avg_features])
                predicted_idx = prediction[0]
                predicted_word = self.inverse_label_mapping.get(predicted_idx, "Unknown")
            else:
                predicted_word = "Model not loaded or incorrect feature length"
            self.text_output += predicted_word + " "
            self.output_text.delete(1.0, "end")
            self.output_text.insert(1.0, f"Translating: {self.text_output}")
        else:
            self.output_text.delete(1.0, "end")
            self.output_text.insert(1.0, "Translating: No valid data detected")

    def test_mode(self):
        if self.model_mixed is None:
            messagebox.showerror("Error", "Please select a model file first!")
            return
        self.running = True
        self.output_text.delete(1.0, "end")
        self.output_text.insert(1.0, "Testing: (Press 'q' to exit)")
        cv2.namedWindow("Test Mode", cv2.WINDOW_AUTOSIZE)
        landmarks_history = []
        motion_detected = False
        start_frame = 0
        expected_length = None
        min_frames = 30

        while self.running:
            with self.lock:
                if self.frame_data is None:
                    continue
                frame = self.frame_data.copy()
            landmarks, frame_with_skeleton, bounding_boxes, hands_detected = self.extract_landmarks(frame)
            if hands_detected and landmarks:
                if expected_length is None:
                    expected_length = len(landmarks[0])
                if len(landmarks[0]) == expected_length:
                    if not motion_detected and len(landmarks_history) > 1:
                        prev_data = np.array(landmarks_history[-1])
                        curr_data = np.array(landmarks[0])
                        if len(curr_data) == len(prev_data) and np.linalg.norm(curr_data - prev_data) > 0.01:
                            motion_detected = True
                            start_frame = len(landmarks_history)
                    landmarks_history.append(landmarks[0])
                    if motion_detected and len(landmarks_history) > start_frame + 1 and len(landmarks_history) >= min_frames:
                        curr_data = np.array(landmarks[0])
                        prev_data = np.array(landmarks_history[-2])
                        if len(curr_data) == len(prev_data) and np.linalg.norm(curr_data - prev_data) < 0.002:
                            end_frame = len(landmarks_history) - 1
                            break
            predicted_word = "Unknown"
            if len(landmarks_history) > start_frame and self.model_mixed:
                frame_features = landmarks_history[start_frame:]
                expected_length = len(frame_features[0])
                frame_features = [f for f in frame_features if len(f) == expected_length]
                if frame_features:
                    avg_features = np.mean(frame_features, axis=0)
                    if len(avg_features) == 88:
                        prediction = self.model_mixed.predict([avg_features])
                        predicted_idx = prediction[0]
                        predicted_word = self.inverse_label_mapping.get(predicted_idx, "Unknown")
                    else:
                        predicted_word = "Invalid feature length"
            cv2.putText(
                frame_with_skeleton,
                "Press 'q' to exit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame_with_skeleton,
                f"Detected: {predicted_word}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Test Mode", frame_with_skeleton)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                cv2.destroyWindow("Test Mode")
                break
        self.output_text.delete(1.0, "end")
        self.output_text.insert(1.0, "Testing stopped")

    def test_movement_detection(self):
        video_path = filedialog.askopenfilename(
            title="Select Sample Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not video_path:
            messagebox.showerror("Error", "No video file selected!")
            return

        self.output_text.delete(1.0, "end")
        self.output_text.insert(1.0, "Testing Movement Detection...")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, 60)
        landmarks_history = []
        motion_detected = False
        start_frame = 0
        end_frame = None
        expected_length = None
        min_frames = 30
        prev_landmarks = None
        smoothing_factor = 0.7

        cv2.namedWindow("Movement Detection Test", cv2.WINDOW_AUTOSIZE)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, frame_with_skeleton, _, hands_detected = self.extract_landmarks(frame, draw_skeleton=True)
            if hands_detected and landmarks:
                if expected_length is None:
                    expected_length = len(landmarks[0])
                if len(landmarks[0]) == expected_length:
                    if prev_landmarks is not None:
                        smoothed_landmarks = smoothing_factor * prev_landmarks + (1 - smoothing_factor) * np.array(landmarks[0])
                        landmarks[0] = smoothed_landmarks.tolist()
                    if not motion_detected and len(landmarks_history) > 1:
                        prev_data = np.array(landmarks_history[-1])
                        curr_data = np.array(landmarks[0])
                        if len(curr_data) == len(prev_data) and np.linalg.norm(curr_data - prev_data) > 0.01:
                            motion_detected = True
                            start_frame = len(landmarks_history)
                            cv2.putText(frame_with_skeleton, "Movement Detected: Start", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    landmarks_history.append(landmarks[0])
                    if motion_detected and len(landmarks_history) > 1 and len(landmarks_history) >= min_frames:
                        curr_data = np.array(landmarks[0])
                        prev_data = np.array(landmarks_history[-2])
                        if len(curr_data) == len(prev_data) and np.linalg.norm(curr_data - prev_data) < 0.002:
                            end_frame = len(landmarks_history) - 1
                            cv2.putText(frame_with_skeleton, "Movement Detected: End", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                            break
                    prev_landmarks = np.array(landmarks[0])

            cv2.imshow("Movement Detection Test", frame_with_skeleton)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow("Movement Detection Test")
        self.output_text.delete(1.0, "end")
        self.output_text.insert(1.0, f"Testing completed. Movement detected from frame {start_frame} to {end_frame if end_frame else 'end'}")

    def translate_loop(self):
        if not self.running:
            return
        with self.lock:
            if self.frame_data is None:
                self.frame.after(50, self.translate_loop)
                return
            frame = self.frame_data.copy()
        current_time = time.time()
        landmarks, _, _, hands_detected = self.extract_landmarks(frame, draw_skeleton=False)

        if not hands_detected:
            if not self.is_waiting:
                self.is_waiting = True
                self.wait_start_time = current_time
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, "Translating: Waiting for hands (3 seconds)...")
            elif current_time - self.wait_start_time >= self.wait_duration:
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, "Translating: Waiting mode...")
            self.frame.after(50, self.translate_loop)
            return
        else:
            if self.is_waiting:
                self.is_waiting = False
                self.wait_start_time = None
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, "Translating: Hands detected, resuming...")

        if landmarks and current_time - self.last_update_time >= self.interval:
            if len(landmarks) > 0:
                avg_features = np.mean(landmarks, axis=0)
                if self.model_mixed and len(avg_features) == 88:
                    prediction = self.model_mixed.predict([avg_features])
                    predicted_idx = prediction[0]
                    predicted_word = self.inverse_label_mapping.get(predicted_idx, "Unknown")
                    self.text_output += predicted_word + " "
                else:
                    self.text_output += "[Model not loaded or incorrect feature length] "
            self.output_text.delete(1.0, "end")
            self.output_text.insert(1.0, f"Translating: {self.text_output}")
            self.last_update_time = current_time
        self.frame.after(50, self.translate_loop)

    def destroy(self):
        self.running = False
        self.recording = False
        if hasattr(self, 'video_thread'):
            self.video_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.frame.destroy()

class TranslateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator Application")
        self.root.geometry("1280x720")
        self.root.configure(bg="#D3D3D3")
        self.current_frame = None
        self.setup_gui()

    def setup_gui(self):
        self.content_frame = Frame(self.root, bg="#D3D3D3")
        self.content_frame.pack(fill="both", expand=True)
        footer_label = Label(self.root, text="BEKFURR INC 2025", font=("Arial", 10), bg="#D3D3D3", anchor="se")
        footer_label.pack(side="bottom", fill="x", pady=5)
        self.show_realtime_translator()

    def clear_content(self):
        if self.current_frame:
            self.current_frame.destroy()
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def show_realtime_translator(self):
        self.clear_content()
        self.current_frame = RealtimeTranslator(self.content_frame)

if __name__ == "__main__":
    root = Tk()
    app = TranslateApp(root)
    root.mainloop()