# Importation des bibliothèques nécessaires
from ultralytics import YOLO  # Pour le modèle de détection YOLOv8
import supervision as sv  # Librairie de suivi (tracking)
import pickle  # Pour la sauvegarde/chargement des objets Python
import os
import numpy as np
import pandas as pd
import cv2  # OpenCV pour le traitement d’image
import sys

# Ajout du chemin du dossier parent pour accéder aux fonctions utilitaires
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position  # Fonctions utilitaires personnalisées

# Classe principale pour la détection et le suivi des objets
class Tracker:
    def __init__(self, model_path):
        # Chargement du modèle YOLO
        self.model = YOLO(model_path)
        # Initialisation du tracker ByteTrack
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        """
        Ajoute une position (centre ou position des pieds) à chaque objet suivi.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)  # Position du centre pour le ballon
                    else:
                        position = get_foot_position(bbox)  # Position des pieds pour les joueurs/arbitres
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpole les positions manquantes du ballon (utile quand la détection est intermittente).
        """
        # Extraction des bounding boxes du ballon
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolation des valeurs manquantes
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()  # Remplissage arrière

        # Conversion vers le format original
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        """
        Effectue la détection d'objets sur une liste d'images par batch.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Effectue le suivi des objets sur une séquence de frames.
        Permet également de charger/sauvegarder les résultats pour éviter les recalculs.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        # Initialisation des dictionnaires de suivi
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Parcours de chaque frame
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # Mapping id -> nom
            cls_names_inv = {v: k for k, v in cls_names.items()}  # Mapping nom -> id

            # Conversion des détections au format de supervision
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Remplacement des gardiens par des joueurs dans les classes
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Application du suivi avec ByteTrack
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialisation des dictionnaires pour cette frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Traitement des objets suivis (joueurs et arbitres)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox, "class_name": "player"}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox, "class_name": "referee"}

            # Traitement spécifique pour le ballon
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Sauvegarde optionnelle des résultats
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None, class_name=""):
        """
        Dessine une ellipse sur l'image avec une étiquette de classe.
        Utilisé pour les joueurs et les arbitres.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Dessin de l'ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Dessin du rectangle de texte
        rectangle_width = 80
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15
        x1_text = x1_rect + 10

        # Ajout du texte
        cv2.putText(
            frame,
            class_name,
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        return frame

    def draw_traingle(self, frame, bbox, color):
        """
        Dessine un triangle au-dessus d'un objet (par exemple, un joueur avec le ballon ou le ballon lui-même).
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        """
        Dessine toutes les annotations sur les frames de la vidéo :
        - ellipse pour joueurs et arbitres
        - triangle pour le ballon et joueurs en possession du ballon
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Dessin des joueurs
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 100, 255))  # Couleur d’équipe par défaut
                frame = self.draw_ellipse(
                    frame,
                    player["bbox"],
                    color,
                    track_id,
                    class_name=player.get("class_name", "")
                )

                # Triangle si le joueur a le ballon
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Dessin des arbitres
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Dessin du ballon
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames