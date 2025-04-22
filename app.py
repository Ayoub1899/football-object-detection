# app.py

from flask import Flask, render_template, Response, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

import cv2
import time
import threading
import queue
import os
import shutil

from tracker import Tracker  # Classe de suivi personnalisé

# Initialisation Flask
app = Flask(__name__)

# ------------------ CONFIGURATION GLOBALE ------------------

class Config:
    MODEL_PATH = "models/best.pt"
    UPLOAD_FOLDER = "uploads"
    OUTPUT_FOLDER = "processed_videos"
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

    @classmethod
    def get_video_list(cls):
        folders = [cls.UPLOAD_FOLDER]
        videos = []
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
            videos.extend([
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(tuple(cls.ALLOWED_EXTENSIONS))
            ])
        return videos

    @classmethod
    def allowed_file(cls, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS


# Définir les répertoires utilisés
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = Config.OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload


# ------------------ CLASSE DE TRAITEMENT VIDÉO ------------------

class VideoProcessor:
    def __init__(self):
        self.detection_model = YOLO(Config.MODEL_PATH).to('cpu')
        self.tracker = Tracker(Config.MODEL_PATH)
        self.cap = None
        self.video_path = None
        self.processing = False
        self.end_of_video = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        self.last_frame = None
        self.is_video_active = False

        # Threads
        self.detection_thread = None
        self.save_thread = None

        # Files
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.processed_frames = []

        # Export
        self.current_output_path = None
        self.is_saving = False

    def set_video(self, video_path):
        self.stop()

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        self.end_of_video = False
        self.processing = True
        self.processed_frames = []
        self.last_frame = None

        # Nettoyer les queues
        for q in [self.frame_queue, self.result_queue]:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

        # Préparer sortie vidéo
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        self.current_output_path = os.path.join(
            Config.OUTPUT_FOLDER, f"processed_{os.path.basename(video_path)}"
        )

        # Lancer les threads
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        threading.Thread(target=self.read_frames_worker, daemon=True).start()
        self.is_video_active = True

        return self.cap.isOpened()

    def read_frames_worker(self):
        batch = []
        batch_size = 5
        while self.processing and self.cap and self.cap.isOpened():
            if self.frame_queue.qsize() < 20:
                ret, frame = self.cap.read()
                self.current_frame += 1
                if not ret:
                    if batch:
                        self.frame_queue.put(batch.copy())
                    self.end_of_video = True
                    break
                batch.append(frame.copy())
                if len(batch) >= batch_size:
                    self.frame_queue.put(batch.copy())
                    batch = []
            else:
                time.sleep(0.01)
        if batch and self.processing:
            self.frame_queue.put(batch.copy())

    def detection_worker(self):
        while True:
            if not self.processing and self.frame_queue.empty():
                time.sleep(0.1)
                continue
            try:
                batch = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            tracks = self.tracker.get_object_tracks(batch)
            annotated_frames = self.tracker.draw_annotations(batch, tracks)

            for annotated_frame in annotated_frames:
                progress = (self.current_frame / self.total_frames) * 100 if self.total_frames > 0 else 0
                video_name = os.path.basename(self.video_path)
                cv2.putText(annotated_frame, f"Video: {video_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Progression: {progress:.1f}%", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                self.result_queue.put(annotated_frame)

            self.processed_frames.extend(annotated_frames)
            self.frame_queue.task_done()

            if self.end_of_video and self.frame_queue.empty():
                if not self.is_saving and self.processed_frames:
                    self.save_thread = threading.Thread(target=self.save_video)
                    self.save_thread.start()
                break

    def process_video(self):
        if not self.processing or not self.cap or not self.cap.isOpened() or not self.is_video_active:
            return None
        try:
            frame = self.result_queue.get_nowait()
            self.result_queue.task_done()
            self.last_frame = frame
            return frame
        except queue.Empty:
            return self.last_frame

    def save_video(self):
        if self.is_saving or not self.processed_frames:
            return
        self.is_saving = True
        try:
            h, w = self.processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(self.current_output_path, fourcc, self.fps, (w, h))
            for frame in self.processed_frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"Vidéo sauvegardée à: {self.current_output_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
        finally:
            self.is_saving = False

    def stop(self):
        self.processing = False
        self.is_video_active = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_output_path(self):
        return self.current_output_path


video_processor = VideoProcessor()


# ------------------ ROUTES FLASK ------------------

@app.route('/')
def index():
    processed_videos = os.listdir(Config.OUTPUT_FOLDER) if os.path.exists(Config.OUTPUT_FOLDER) else []
    processed_videos = [f for f in processed_videos if f.lower().endswith(tuple(Config.ALLOWED_EXTENSIONS))]
    status = "completed" if video_processor.end_of_video else "processing" if video_processor.processing else "idle"
    return render_template("index.html", processed_videos=processed_videos,
                           status=status, is_saving=video_processor.is_saving,
                           is_video_active=video_processor.is_video_active)

@app.route('/home')
def home():
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        prev_time = time.time()
        while True:
            frame = video_processor.process_video()
            if frame is None:
                time.sleep(0.1)
                continue
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_video', methods=['POST'])
def select_video():
    video_name = request.form.get('video')
    if video_name:
        video_path = os.path.join(Config.UPLOAD_FOLDER, video_name)
        if os.path.exists(video_path):
            success = video_processor.set_video(video_path)
            return {"status": "success" if success else "error"}
    return {"status": "error", "message": "Vidéo non trouvée"}

@app.route('/stop_video', methods=['POST'])
def stop_video():
    video_processor.stop()
    return {"status": "success"}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename and Config.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(path)
        video_processor.set_video(path)
    return redirect(url_for('index'))

@app.route('/get_status')
def get_status():
    if video_processor.is_saving:
        return {"status": "saving"}
    elif video_processor.end_of_video:
        return {"status": "completed"}
    elif video_processor.processing:
        return {"status": "processing"}
    else:
        return {"status": "idle"}

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(Config.OUTPUT_FOLDER, filename)
    return send_file(path, as_attachment=True) if os.path.exists(path) else ("Fichier non trouvé", 404)

@app.route('/delete_processed/<filename>')
def delete_processed(filename):
    os.remove(os.path.join(Config.OUTPUT_FOLDER, filename))
    return redirect(url_for('index'))

@app.route('/delete_uploaded/<filename>')
def delete_uploaded(filename):
    os.remove(os.path.join(Config.UPLOAD_FOLDER, filename))
    return redirect(url_for('index'))

@app.route('/delete_all_uploads', methods=['POST'])
def delete_all_uploads():
    shutil.rmtree(Config.UPLOAD_FOLDER, ignore_errors=True)
    os.makedirs(Config.UPLOAD_FOLDER)
    return {"status": "success"}


# ------------------ LANCEMENT ------------------

if __name__ == '__main__':
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    finally:
        video_processor.stop()
