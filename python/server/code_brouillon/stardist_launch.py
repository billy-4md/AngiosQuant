from __future__ import print_function, unicode_literals, absolute_import, division

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tifffile import imread, imwrite
from aicspylibczi import CziFile
from csbdeep.utils import normalize
from csbdeep.io import save_tiff_imagej_compatible
from stardist.models import StarDist2D
from stardist import random_label_cmap
from xml.etree import ElementTree as ET

np.random.seed(6)
lbl_cmap = random_label_cmap()

# Chemin vers le fichier CZI
czi_file_path = '/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/uploads/cellules seules 40X-1.czi'

# Charger l'image CZI et extraire les données
czi_file = CziFile(czi_file_path)

# # Lire les métadonnées XML brutes
# metadata_xml = czi_file.meta
# # Parser le XML des métadonnées
# tree = ET.ElementTree(ET.fromstring(metadata_xml))
# root = tree.getroot()

# # Exemple pour accéder à la dimension Z
# # Vous devrez adapter la recherche en fonction de la structure de vos métadonnées
# dimensions = root.find('.//Dimensions')
# size_z = int(dimensions.find('.//Z').text)

# print(f"Nombre de tranches Z: {size_z}")

# Charger le modèle pré-entraîné StarDist 2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# Préparer un conteneur pour les étiquettes de toutes les tranches
labels_all_slices = []

# Lire et traiter chaque tranche Z
for z in range(371):
    czi_image = czi_file.read_mosaic(C=1, Z= z, scale_factor=1.0)
    image = np.squeeze(np.asarray(czi_image))
    print(image.shape)
    img_slice = normalize(image, 1, 99.8, axis=(0,1))
    labels, details = model.predict_instances(img_slice)
    labels_all_slices.append(labels)

labels_all_slices = np.stack(labels_all_slices, axis=0)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(img_slice, cmap='gray')  # Afficher la dernière tranche traitée
ax[0].set_title('Last processed slice from the 3D image')
ax[0].axis('off')
ax[1].imshow(labels, cmap=lbl_cmap, alpha=0.5)  # Afficher les étiquettes de la dernière tranche traitée
ax[1].set_title('Predicted labels for the last processed slice')
ax[1].axis('off')
plt.tight_layout()
plt.show()

# Enregistrer les étiquettes au format TIFF, si nécessaire
save_tiff_imagej_compatible('labels_from_3D_czi.tif', labels_all_slices, axes='ZYX')

