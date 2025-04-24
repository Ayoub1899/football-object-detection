document.addEventListener('DOMContentLoaded', function() {
    const videoList = document.getElementById('videoList');
    const startProcessingBtn = document.getElementById('startProcessingBtn');
    const stopProcessingBtn = document.getElementById('stopProcessingBtn');
    const processingStatus = document.getElementById('processingStatus');
    const videoFeed = document.getElementById('videoFeed');
    const deleteAllUploadsBtn = document.getElementById('deleteAllUploadsBtn');

    videoList.addEventListener('change', function() {
        startProcessingBtn.disabled = !this.value;
    });

    startProcessingBtn.addEventListener('click', function() {
        const selectedVideo = videoList.value;
        if (selectedVideo) {
            fetch('/select_video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `video=${selectedVideo}`
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    stopProcessingBtn.disabled = false;
                    startProcessingBtn.disabled = true;
                    processingStatus.textContent = 'Statut: Démarrage du traitement...';
                    // Le flux vidéo devrait commencer à s'afficher automatiquement via l'attribut src de l'image
                } else {
                    alert(data.message || 'Erreur lors de la sélection de la vidéo.');
                }
            })
            .catch(error => {
                console.error('Erreur:', error);
                alert('Une erreur s\'est produite.');
            });
        }
    });

    stopProcessingBtn.addEventListener('click', function() {
        fetch('/stop_video', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                stopProcessingBtn.disabled = true;
                startProcessingBtn.disabled = false;
                processingStatus.textContent = 'Statut: Traitement arrêté.';
                videoFeed.src = "{{ url_for('video_feed') }}"; // Reset le flux (peut-être pas nécessaire)
            } else {
                alert('Erreur lors de l\'arrêt du traitement.');
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            alert('Une erreur s\'est produite.');
        });
    });

    const deleteButtons = document.querySelectorAll('.delete-btn');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function() {
            const filename = this.dataset.filename;
            const isUploaded = this.parentNode.parentNode.parentNode.classList.contains('upload-section');
            const route = isUploaded ? '/delete_uploaded/' : '/delete_processed/';
            if (confirm(`Êtes-vous sûr de vouloir supprimer ${filename} ?`)) {
                fetch(route + filename)
                .then(response => response.text())
                .then(() => {
                    window.location.reload();
                })
                .catch(error => {
                    console.error('Erreur:', error);
                    alert('Erreur lors de la suppression du fichier.');
                });
            }
        });
    });

    deleteAllUploadsBtn.addEventListener('click', function() {
        if (confirm("Êtes-vous sûr de vouloir supprimer toutes les vidéos importées ?")) {
            fetch('/delete_all_uploads', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    window.location.reload();
                } else {
                    alert("Erreur lors de la suppression des vidéos importées.");
                }
            })
            .catch(error => {
                console.error("Erreur:", error);
                alert("Une erreur s'est produite lors de la suppression des vidéos importées.");
            });
        }
    });

    function updateStatus() {
        fetch('/get_status')
        .then(response => response.json())
        .then(data => {
            processingStatus.textContent = `Statut: ${data.status}${data.status === 'saving' ? ' (Sauvegarde en cours...)' : ''}`;
            if (data.status === 'completed') {
                stopProcessingBtn.disabled = true;
                startProcessingBtn.disabled = false;
                // Vous pouvez ici afficher une notification ou mettre à jour l'interface
            }
        })
        .catch(error => {
            console.error('Erreur lors de la récupération du statut:', error);
        });
    }

    setInterval(updateStatus, 1000); // Vérifier le statut toutes les secondes
});