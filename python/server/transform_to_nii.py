from imio import load, save

# img = load.load_any("D:\\Users\\MFE\\Xavier\\8IFnew\\image1_processed.tif")
# save.to_nii(img, "D:\\Users\\MFE\\Xavier\\8IFnew\\img1_nifti.nii" )


# # Charger l'image NIfTI
# img = load.load_any("D:\\Users\\MFE\\Xavier\\8IFnew\\medseg_mask_v3.nii")

# # Enregistrer l'image en format TIFF
# save.to_tiff(img, "D:\\Users\\MFE\\Xavier\\8IFnew\\medseg_mask_hand.tif")

import numpy as np
import tifffile as tiff
from sklearn.metrics import jaccard_score
#from skimage.metrics import dice

model = ["nuclei", "cyto", "cyto3"]
model_star = ["versatile", "paper_dsb2018"]
diam = ["30", "60", "90"]

# Charger les images de segmentation
seg1 = tiff.imread('D:\\Users\\MFE\\Xavier\\8IFnew\\medseg_mask_hand.tif')

for model_name in model:
    for diam_value in diam:
        seg2 = tiff.imread(f'D:\\Users\\MFE\\Xavier\\Result_Rapport2\\CellPose_pred_2D_{model_name}_diam{diam_value}.tif')
        print(seg2.dtype)
        print(seg2.shape)

        # Assurer que les images sont binaires
        seg1_binary = seg1 > 0
        seg2_binary = seg2 > 0

        # Aplatir les images pour les métriques
        seg1_flat = seg1_binary.ravel()
        seg2_flat = seg2_binary.ravel()

        # Calculer l'Indice de Jaccard
        jaccard = jaccard_score(seg1_flat, seg2_flat)

        # Calculer le Dice Coefficient
        #diceco = dice(seg1_flat, seg2_flat)

        print(f'Jaccard Index for model: {model_name} diameter: {diam_value}: {jaccard}')
        #print(f'Dice Coefficient: {diceco}')

seg1 = tiff.imread('D:\\Users\\MFE\\Xavier\\8IFnew\\medseg_mask_hand.tif')
print(seg1.dtype)

for model_name in model_star:
    seg2 = tiff.imread(f'D:\\Users\\MFE\\Xavier\\Result_Rapport2\\label_2D_StarDist_{model_name}.tif')
    #seg2 = seg2.astype(np.float64)

    # Assurer que les images sont binaires
    seg1_binary = (seg1 > 0)
    seg2_binary = (seg2 > 0)

    # Aplatir les images pour les métriques
    seg1_flat = seg1_binary.ravel()
    seg2_flat = seg2_binary.ravel()

    # Calculer l'Indice de seg1_flat
    jaccard = jaccard_score(seg1_flat, seg2_flat)

    # Calculer le Dice Coefficient
    #diceco = dice(seg1_flat, seg2_flat)

    print(f'Jaccard Index for model: {model_name}: {jaccard}')
    #print(f'Dice Coefficient: {diceco}')

