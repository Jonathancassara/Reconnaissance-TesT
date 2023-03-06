import cv2
import numpy as np
import tensorflow as tf

# Charger le modèle de reconnaissance de personnes entraîné
model = tf.keras.models.load_model('model.h5')

# Charger l'image temoin
template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)

# Créer le détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialiser le score à 0
score = 0

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    # Lire l'image de la webcam
    ret, frame = cap.read()
    
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Pour chaque visage détecté, effectuer la reconnaissance de personne
    for (x, y, w, h) in faces:
        # Extraire la région du visage
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Redimensionner l'image de visage pour correspondre à la taille de l'image temoin
        roi_gray_resized = cv2.resize(roi_gray, template.shape[::-1])
        
        # Normaliser les valeurs de pixels entre 0 et 1
        roi_gray_norm = roi_gray_resized / 255.0
        
        # Ajouter une dimension de lot (batch) pour l'entrée du modèle
        roi_gray_norm = np.expand_dims(roi_gray_norm, axis=0)
        roi_gray_norm = np.expand_dims(roi_gray_norm, axis=3)
        
        # Effectuer la prédiction de personne
        pred = model.predict(roi_gray_norm)[0]
        
        # Si la prédiction est correcte, augmenter le score
        if np.argmax(pred) == 0:
            score += 1
        
        # Dessiner un rectangle autour du visage et afficher le score
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Score: ' + str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # Afficher l'image
    cv2.imshow('frame', frame)
    
    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer la fenêtre d'affichage
cap.release()
cv2.destroyAllWindows()
