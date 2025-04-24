# ⚽ Application Flask - Détection en temps réel de scènes de football

Cette application web permet de détecter en **temps réel** les **joueurs**, **arbitres**, **gardiens** et le **ballon** dans une vidéo de match de football grâce à un modèle **YOLOv11**. L’utilisateur peut uploader une vidéo, suivre le traitement en direct, stopper ou télécharger la vidéo annotée.

---

## 🚀 Installation

1. **Cloner ce dépôt :**
   ```bash
   git clone https://github.com/Ayoub189/football-object-detection.git
   cd projet-foot-detection
2. **Créer un environnement virtuel (optionnel mais recommandé) :**
   ```bash
   python -m venv venv
   source venv/bin/activate  # sous Linux/macOS
   venv\Scripts\activate     # sous Windows
   ```
3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
   ```
**Prérequis**
- Python 3.8+

**Utilisation :**
1. Lancer l'application Flask :
  ```bash
  python app.py
```
2. Ouvrir l'application dans votre navigateur :
  ```bash
http://127.0.0.1:5000
```
3. Fonctionnalités principales :

- Upload de vidéo (.mp4, .avi, .mov, .mkv)

- Détection en temps réel via un fichier

- Visualisation du flux avec les annotations

- Téléchargement des vidéos traitées

- Historique des vidéos traitées
