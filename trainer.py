import os
import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, Frame, Button, Label, messagebox, filedialog
import pickle
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, parent):
        self.frame = Frame(parent, bg="#D3D3D3")
        self.frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)
        self.selected_folder = None
        self.setup_gui()

    def setup_gui(self):
        Label(self.frame, text="Data Processor", font=("Arial", 16, "bold"), bg="#D3D3D3").pack(pady=10)
        Button(self.frame, text="Select Data Folder", command=self.select_folder, bg="#FFFFFF", font=("Arial", 12), relief="flat", borderwidth=2).pack(pady=5)
        self.folder_label = Label(self.frame, text="No folder selected", font=("Arial", 12), bg="#D3D3D3")
        self.folder_label.pack(pady=5)
        Button(self.frame, text="Process Data", command=self.process_data, bg="#FFFFFF", font=("Arial", 12), relief="flat", borderwidth=2).pack(pady=5)

    def select_folder(self):
        self.selected_folder = filedialog.askdirectory(title="Select Data Folder")
        if self.selected_folder:
            self.folder_label.config(text=f"Selected: {self.selected_folder}")
        else:
            self.folder_label.config(text="No folder selected")

    def detect_hand_and_elbow_movement(self, video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, 60)
        landmarks_history = []
        motion_detected = False
        start_frame = None
        end_frame = None
        expected_length = None
        min_frames = 30
        prev_landmarks = None
        smoothing_factor = 0.7

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.5, beta=15)

            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)

            data_aux = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
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

            if len(data_aux) < 88:
                data_aux.extend([0.0] * (88 - len(data_aux)))

            if expected_length is None and data_aux:
                expected_length = len(data_aux)
            if len(data_aux) == expected_length:
                if prev_landmarks is not None:
                    smoothed_landmarks = smoothing_factor * prev_landmarks + (1 - smoothing_factor) * np.array(data_aux)
                    data_aux = smoothed_landmarks.tolist()
                landmarks_history.append(data_aux)
                if len(landmarks_history) > 1 and len(landmarks_history) >= min_frames:
                    prev_data = np.array(landmarks_history[-2])
                    curr_data = np.array(data_aux)
                    if len(curr_data) == len(prev_data):
                        diff = np.linalg.norm(curr_data - prev_data)
                        if diff > 0.01:
                            if not motion_detected:
                                start_frame = len(landmarks_history) - 1
                                motion_detected = True
                        elif motion_detected and diff < 0.002:
                            end_frame = len(landmarks_history) - 1
                            break
                prev_landmarks = np.array(data_aux)
            else:
                if motion_detected and end_frame is None and len(landmarks_history) >= min_frames:
                    end_frame = len(landmarks_history) - 1
                    break

            logging.debug(f"Frame processed for {video_path}. Landmarks detected: {len(data_aux)}")

        cap.release()
        if not motion_detected:
            if landmarks_history:
                start_frame = 0
                end_frame = len(landmarks_history) - 1
            else:
                landmarks_history = [np.zeros(88).tolist()]
                start_frame = 0
                end_frame = 0
                logging.warning(f"No landmarks detected in video: {video_path}, using default zero features.")

        return start_frame, end_frame, landmarks_history

    def process_data(self):
        if not self.selected_folder:
            messagebox.showerror("Error", "Please select a data folder first!")
            return

        json_path = os.path.join(self.selected_folder, "words.json")

        if not os.path.exists(json_path):
            video_files = [f for f in os.listdir(self.selected_folder) if f.endswith(".mp4")]
            if not video_files:
                messagebox.showerror("Error", "No .mp4 files found in the selected folder!")
                return
            words_data = [{"word_uz": f.replace(".mp4", ""), "video": f} for f in video_files]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(words_data, f, ensure_ascii=False, indent=2)
            logging.info(f"words.json created: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                words_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Error reading JSON file: {str(e)}")
            return

        data = []
        labels = []
        class_names = []

        for idx, item in enumerate(words_data):
            word = item["word_uz"]
            video_path = os.path.join(self.selected_folder, item["video"])
            if not os.path.exists(video_path):
                logging.warning(f"Video file not found: {video_path}")
                continue

            logging.info(f"Processing video: {video_path} for class: {word}")
            start_frame, end_frame, landmarks_history = self.detect_hand_and_elbow_movement(video_path)

            if landmarks_history:
                frame_features = landmarks_history[start_frame:end_frame + 1] if start_frame is not None and end_frame is not None else landmarks_history
                if frame_features:
                    expected_length = len(frame_features[0])
                    frame_features = [f for f in frame_features if len(f) == expected_length]
                    if len(frame_features) == 0:
                        logging.warning(f"No consistent features extracted from video: {video_path}, using default zero features.")
                        frame_features = [np.zeros(88).tolist()]
                    avg_features = np.mean(frame_features, axis=0)
                    data.append(avg_features)
                    labels.append(word)
                    class_names.append(word)
                    logging.info(f"Successfully processed video: {video_path}, class: {word}")
                else:
                    logging.warning(f"No valid features extracted from video: {video_path}, using default zero features.")
                    data.append(np.zeros(88).tolist())
                    labels.append(word)
                    class_names.append(word)
                    logging.info(f"Added default features for video: {video_path}, class: {word}, label: {word}")

        if not data:
            messagebox.showerror("Error", "No valid data processed from videos!")
            return

        logging.info(f"Processed {len(data)} samples with {len(set(labels))} unique classes: {class_names}")
        with open('data_mixed.pickle', 'wb') as f:
            pickle.dump({'data': data, 'labels': labels, 'class_names': class_names}, f)
        messagebox.showinfo("Success", "Data processed and saved as data_mixed.pickle")

    def destroy(self):
        self.frame.destroy()

class ModelTrainer:
    def __init__(self, parent):
        self.frame = Frame(parent, bg="#D3D3D3")
        self.frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.selected_pickle = None
        self.setup_gui()

    def setup_gui(self):
        Label(self.frame, text="Model Trainer", font=("Arial", 16, "bold"), bg="#D3D3D3").pack(pady=10)
        Button(self.frame, text="Select Pickle File", command=self.select_pickle_file, bg="#FFFFFF", font=("Arial", 12), relief="flat", borderwidth=2).pack(pady=5)
        self.pickle_label = Label(self.frame, text="No pickle file selected", font=("Arial", 12), bg="#D3D3D3")
        self.pickle_label.pack(pady=5)
        Button(self.frame, text="Train Model", command=self.train_model, bg="#FFFFFF", font=("Arial", 12), relief="flat", borderwidth=2).pack(pady=5)

    def select_pickle_file(self):
        self.selected_pickle = filedialog.askopenfilename(
            title="Select Pickle File",
            filetypes=[("Pickle files", "*.pickle")]
        )
        if self.selected_pickle:
            self.pickle_label.config(text=f"Selected: {self.selected_pickle}")
        else:
            self.pickle_label.config(text="No pickle file selected")

    def train_model(self):
        if not self.selected_pickle:
            messagebox.showerror("Error", "Please select a pickle file first!")
            return

        try:
            data_dict = pickle.load(open(self.selected_pickle, 'rb'))
            data = data_dict['data']
            labels = data_dict['labels']
            if not data:
                messagebox.showerror("Error", "No data found in the pickle file!")
                return

            unique_classes = len(set(labels))
            if unique_classes < 2:
                messagebox.showwarning("Warning", f"Only {unique_classes} class found! Training with limited data might not be effective.")
                if unique_classes == 0:
                    messagebox.showerror("Error", "No classes found!")
                    return

            feature_lengths = [len(d) for d in data]
            if not feature_lengths:
                messagebox.showerror("Error", "No valid feature lengths found in data!")
                return
            most_common_length = max(set(feature_lengths), key=feature_lengths.count)
            logging.info(f"Most common feature length: {most_common_length}")

            filtered_data = [d for d in data if len(d) == most_common_length]
            filtered_labels = [labels[i] for i, d in enumerate(data) if len(d) == most_common_length]

            if not filtered_data:
                messagebox.showerror("Error", "No valid data for training after filtering!")
                return

            data = np.asarray(filtered_data)
            labels = np.asarray(filtered_labels)

            le = LabelEncoder()
            labels_encoded = le.fit_transform(labels)
            label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

            x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.1, shuffle=True)
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(x_train, y_train)
            y_predict = model.predict(x_test)
            score = accuracy_score(y_predict, y_test)
            logging.info(f'Hand + Elbow: {score * 100:.2f}% of samples classified correctly!')

            with open('model_mixed.p', 'wb') as f:
                pickle.dump({'model': model, 'label_mapping': label_mapping}, f)
            messagebox.showinfo("Success", f"Model trained with accuracy {score * 100:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def destroy(self):
        self.frame.destroy()

class TrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Training Application")
        self.root.geometry("1280x720")
        self.root.configure(bg="#D3D3D3")
        self.current_frame = None
        self.setup_gui()

    def setup_gui(self):
        nav_frame = Frame(self.root, bg="#C0C0C0")
        nav_frame.pack(fill="x", padx=10, pady=10)
        Button(nav_frame, text="Data Processor", command=self.show_data_processor, bg="#FFFFFF", font=("Arial", 12), relief="flat", borderwidth=2).pack(side="left", padx=5)
        Button(nav_frame, text="Model Trainer", command=self.show_model_trainer, bg="#FFFFFF", font=("Arial", 12), relief="flat", borderwidth=2).pack(side="left", padx=5)
        self.content_frame = Frame(self.root, bg="#D3D3D3")
        self.content_frame.pack(fill="both", expand=True)
        footer_label = Label(self.root, text="BEKFURR INC 2025", font=("Arial", 10), bg="#D3D3D3", anchor="se")
        footer_label.pack(side="bottom", fill="x", pady=5)
        self.show_data_processor()

    def clear_content(self):
        if self.current_frame:
            self.current_frame.destroy()
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def show_data_processor(self):
        self.clear_content()
        self.current_frame = DataProcessor(self.content_frame)

    def show_model_trainer(self):
        self.clear_content()
        self.current_frame = ModelTrainer(self.content_frame)

if __name__ == "__main__":
    root = Tk()
    app = TrainApp(root)
    root.mainloop()