from PIL import Image
import numpy as np

def appliquer_masque(image_path, masque_path, resultat_path):
    # Charger l'image originale et le masque
    image = Image.open(image_path).convert("RGBA")
    masque = Image.open(masque_path).convert("L")  # Convertir le masque en niveau de gris

    # Convertir les images en tableaux numpy
    image_np = np.array(image)
    masque_np = np.array(masque)

    # Inverser le masque: les parties blanches (255) deviennent noires (0) et vice versa
    # Ainsi, 255 (blanc) dans le masque original signifie "enlever" et deviendra 0 (noir) dans le masque inversé, signifiant "garder"
    masque_inverse = 255 - masque_np

    # Étendre la dimension du masque pour correspondre à celle de l'image RGBA
    masque_inverse = np.stack([masque_inverse] * 3 + [np.ones_like(masque_inverse) * 255], axis=-1)

    # Appliquer le masque inversé à l'image
    resultat_np = np.where(masque_inverse, image_np, 0)  # Remplace par 0 (noir) là où le masque est noir

    # Convertir le tableau résultant en une image PIL
    resultat = Image.fromarray(resultat_np.astype('uint8'), 'RGBA')
    
    # Sauvegarder l'image résultante
    resultat.save(resultat_path)

# Chemins des fichiers
image_path = '/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/hough_transform/1.IF1 merge.tif'
masque_path = '/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/hough_transform/segmentation/1.IF1 1.tif/1.IF1 1_median_filtered_00511_new2.tif'
resultat_path = '/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/hough_transform/result_mask_off_IF1 merge.tif'

# Appeler la fonction
appliquer_masque(image_path, masque_path, resultat_path)
