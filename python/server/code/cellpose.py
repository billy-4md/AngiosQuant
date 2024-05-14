from cellpose import models, io

# Charger votre image 3D
# 'image' devrait être un tableau numpy 3D (z, x, y) ou une liste de tableaux 2D
image = io.imread('path_to_your_3D_image')

# Initialiser le modèle Cellpose
model = models.Cellpose(model_type='cyto', gpu=False)

# Effectuer la segmentation sur l'image 3D
# Assurez-vous de mettre 'do_3D=True' pour les images 3D
# Le paramètre 'anisotropy' pourrait être important si les tailles des voxels dans les différentes dimensions ne sont pas égales
masks, flows, styles, diams = model.eval(image, diameter=0, channels=[0,0], do_3D=True, anisotropy=None)

# Enregistrer les masques segmentés
# Vous pouvez ajuster le chemin et le nom de fichier selon vos besoins
io.masks_flows_to_seg(image, masks, flows, diams, 'output_filename_3D', channels=[0,0])
