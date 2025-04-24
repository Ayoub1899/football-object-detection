from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import time
import threading
import queue
import os
import shutil
from tracker import Tracker

# Initialisation Flask
app = Flask(__name__)


# ------------------ CONFIGURATION GLOBALE ------------------

class Config:
    MODEL_PATH = "models/best.pt"
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

    @classmethod
    def allowed_file(cls, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS


# Définir les répertoires utilisés
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload par fichier


# ------------------ FONCTIONS UTILITAIRES ------------------

def get_folder_size(folder):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def clean_uploads_folder():
    upload_folder = Config.UPLOAD_FOLDER
    if not os.path.exists(upload_folder):
        return

    current_size = get_folder_size(upload_folder)
    if current_size >= Config.MAX_UPLOAD_SIZE:
        print(f"Nettoyage du dossier uploads (taille actuelle: {current_size / 1024 / 1024:.2f} MB)")
        try:
            shutil.rmtree(upload_folder)
            os.makedirs(upload_folder)
            print("Dossier uploads vidé avec succès")
        except Exception as e:
            print(f"Erreur lors du nettoyage du dossier uploads: {e}")


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
        self.is_saving = False


        # Threads
        self.detection_thread = None
        self.save_thread = None

        # Files
        self.frame_queue = queue.Queue(maxsize=60)
        self.result_queue = queue.Queue(maxsize=60)

    def set_video(self, video_path):
        self.stop()
        clean_uploads_folder()  # Vérifier la taille du dossier avant traitement

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        self.end_of_video = False
        self.processing = True
        self.last_frame = None
        self.is_video_active = True

        # Nettoyer les queues
        for q in [self.frame_queue, self.result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Lancer les threads
            # Threads
            self.detection_thread = None
            self.save_thread = None

            # Files
            self.frame_queue = queue.Queue(maxsize=60)
            self.result_queue = queue.Queue(maxsize=60)
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
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Préparer sortie vidéo

        # Lancer les threads
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        threading.Thread(target=self.read_frames_worker, daemon=True).start()
        self.is_video_active = True

        return self.cap.isOpened()

    def stop(self):
        self.processing = False
        self.is_video_active = False
        if self.cap:
            self.cap.release()
            self.cap = None

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
                    self.save_thread = threading.Thread()
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




video_processor = VideoProcessor()


# ------------------ ROUTES FLASK ------------------

@app.route('/')
def index():
    upload_folder = app.config['UPLOAD_FOLDER']
    uploaded_videos = []
    if os.path.exists(upload_folder):
        uploaded_videos = [f for f in os.listdir(upload_folder)
                           if os.path.isfile(os.path.join(upload_folder, f))
                           and f.lower().endswith(tuple(Config.ALLOWED_EXTENSIONS))]

    status = "completed" if video_processor.end_of_video else "processing" if video_processor.processing else "idle"
    return render_template("index.html",
                           uploaded_videos=uploaded_videos,
                           status=status,
                           is_video_active=video_processor.is_video_active)




@app.route('/home')
def home():
    return redirect(url_for('index'))

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


@app.route('/delete_uploaded/<filename>')
def delete_uploaded(filename):
    try:
        os.remove(os.path.join(Config.UPLOAD_FOLDER, filename))
        clean_uploads_folder()  # Vérifier la taille après suppression
    except Exception as e:
        print(f"Erreur suppression fichier: {e}")
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

@app.route('/cleanup', methods=['POST'])
def cleanup():
    video_processor.stop()
    return {'status': 'success'}

@app.route('/get_status')
def get_status():
    if video_processor.is_saving:
        return {"status": "saving"}
    elif video_processor.processing:
        return {"status": "processing"}
    else:
        return {"status": "idle"}

@app.route('/get_full_status')
def get_full_status():
    status = {
        'status': 'idle',
        'current_frame': 0,
        'total_frames': 0
    }

    if video_processor.is_saving:
        status['status'] = 'saving'
    elif video_processor.end_of_video:
        status['status'] = 'completed'
    elif video_processor.processing:
        status['status'] = 'processing'
        status['current_frame'] = video_processor.current_frame
        status['total_frames'] = video_processor.total_frames

    return jsonify(status)


# ... (conservez les autres routes nécessaires comme /video_feed, /stop_video, etc.)

if __name__ == '__main__':
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    try:
        app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    finally:
        video_processor.stop()