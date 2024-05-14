from skimage.transform import resize
import czifile
import numpy as np

# Charger l'image CZI
with czifile.CziFile('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/hough_transform/1.IF1.czi') as czi:
    image_czi = czi.asarray()

# Supposons que img_3d soit votre masque des cercles
# Redimensionner le masque pour qu'il corresponde aux dimensions x et y de l'image CZI
# Note : image_czi peut avoir des dimensions supplémentaires (z, c, t, ...), donc utilisez les dimensions x et y appropriées
masque_redimensionne = resize(img_3d, (image_czi.shape[-2], image_czi.shape[-1]), preserve_range=True, anti_aliasing=True)

# Convertir le masque redimensionné en booléen (True pour les pixels à conserver, False pour ceux à éliminer)
masque_bool = masque_redimensionne > 0  # Ajustez ce seuil en fonction de la façon dont le masque a été généré

# Appliquer le masque à l'image CZI
# Pour chaque tranche/couche dans l'image CZI, appliquez le masque
# Note : Cet exemple suppose une image 3D; ajustez les indices si nécessaire pour votre cas spécifique
for i in range(image_czi.shape[0]):  # Ajustez cette boucle en fonction de la dimension z de votre image CZI
    image_czi[i][masque_bool] = 0  # Définir les pixels masqués à zéro

# Sauvegarder ou utiliser l'image CZI masquée comme nécessaire
# Vous pouvez utiliser la bibliothèque `tifffile` pour sauvegarder l'image résultante si nécessaire
