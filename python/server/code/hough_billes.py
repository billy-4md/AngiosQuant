import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
image_path = '/Volumes/LaCie/Mémoire/Images/New/1.IF1 1.tif'  # Remplacez par le chemin de votre image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Appliquer un flou pour réduire le bruit
blurred_image = cv2.medianBlur(image, 5)

# Appliquer la transformée de Hough pour détecter les cercles
# cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])
# - image: Image en niveaux de gris sur laquelle effectuer la détection.
# - method: Méthode de détection. Dans OpenCV, c'est généralement cv2.HOUGH_GRADIENT.
# - dp: Inverse du ratio de résolution.
# - minDist: Distance minimale entre les centres des cercles détectés.
# - param1: Seuil supérieur pour le détecteur de bords Canny.
# - param2: Seuil pour la détection des centres des cercles.
# - minRadius: Rayon minimum à détecter.
# - maxRadius: Rayon maximum à détecter.
circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, 15,
                            param1=90, param2=75, minRadius=25, maxRadius=150)

# S'assurer que des cercles ont été trouvés
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # dessiner le cercle externe
        cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)
        # dessiner le centre du cercle
        cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 3)

# Afficher l'image avec les cercles détectés
plt.imshow(image, cmap='gray')
plt.title('Detected Circles')
plt.axis('off')
plt.show()
