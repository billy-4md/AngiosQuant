import os
import json
import numpy as np
import pandas as pd

from csbdeep.utils import normalize
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import cdist

from stardist.models import StarDist2D, StarDist3D
from cellpose import models, io, utils, plot


"""-------------SEGMENTATION PROCESS FUNCTIONS----------"""
def do_segmentation_StarDist(isolate_image, SURFACE_THRESHOLD_SUP, SURFACE_THRESHOLD_INF):
    labels_all_slices = []
    data = []
    model = StarDist2D.from_pretrained('2D_versatile_fluo') #Load StarDist model 

    for z in range(isolate_image.shape[0]):
        img_slice = isolate_image[z, :, :]
        img_slice = normalize(img_slice, 1, 99.8)
        filtered_img_slice = img_slice 
        labels, details = model.predict_instances(filtered_img_slice)
        unique_labels = np.unique(labels)

        labels_to_remove = []
        for label_num in unique_labels:
            if label_num == 0: 
                continue

            instance_mask = labels == label_num
            label_surface = np.sum(instance_mask)

            if label_surface > SURFACE_THRESHOLD_SUP or label_surface < SURFACE_THRESHOLD_INF:
                labels[instance_mask] = 0
                labels_to_remove.append(label_num)

        unique_labels = np.array([label for label in unique_labels if label not in labels_to_remove])
        for label in unique_labels:
            if label == 0:  
                continue
            center = center_of_mass(labels == label)
            data.append({'Layer': z, 'Label': label, 'Center X': center[0], 'Center Y': center[1]})

        labels_all_slices.append(labels)
        df = pd.DataFrame(data)

    return reassign_labels(labels_all_slices, df)

def do_segmentation_CellPose(isolate_image, SURFACE_THRESHOLD_SUP, SURFACE_THRESHOLD_INF, diam_value):
    labels_all_slices = []
    data = []
    model = models.Cellpose(gpu=False, model_type="cyto")

    for z in range(isolate_image.shape[0]):
        filtered_img_slice = isolate_image[z, :, :]

        # Evaluate the model with CellPose
        masks, flows, styles, diams = model.eval(filtered_img_slice, diameter=diam_value, channels=[0,0])

        # # Visualize the segmentation results
        # fig = plt.figure(figsize=(8, 6))
        # plot.show_segmentation(fig, filtered_img_slice, masks, flows[0], channels=[0,0])
        # plt.tight_layout()
        # plt.show()

        # Analyze and filter the masks based on surface threshold
        masks = masks.astype(np.uint16)
        unique_labels = np.unique(masks)
        for label_num in unique_labels:
            if label_num == 0:
                continue

            instance_mask = masks == label_num
            label_surface = np.sum(instance_mask)
            if label_surface > SURFACE_THRESHOLD_SUP or label_surface < SURFACE_THRESHOLD_INF:
                masks[instance_mask] = 0  # Remove the label

        # Store data for each label that meets the criteria
        for label in np.unique(masks):
            if label == 0:
                continue
            center = center_of_mass(masks == label)
            data.append({'Layer': z, 'Label': label, 'Center X': center[0], 'Center Y': center[1]})

        labels_all_slices.append(masks)
        df = pd.DataFrame(data)

    return reassign_labels(labels_all_slices, df)
   
def do_segmentation_CellPose_phalo(isolate_image, SURFACE_THRESHOLD_SUP, SURFACE_THRESHOLD_INF, diam_value):
    labels_all_slices = []
    data = []
    model = models.Cellpose(gpu=False, model_type="cyto3")

    for z in range(isolate_image.shape[0]):
        filtered_img_slice = isolate_image[z, :, :]

        # Evaluate the model with CellPose
        masks, flows, styles, diams = model.eval(filtered_img_slice, diameter=diam_value, channels=[0,0])

        # # Visualize the segmentation results
        # fig = plt.figure(figsize=(8, 6))
        # plot.show_segmentation(fig, filtered_img_slice, masks, flows[0], channels=[0,0])
        # plt.tight_layout()
        # plt.show()

        # Analyze and filter the masks based on surface threshold
        masks = masks.astype(np.uint16)
        unique_labels = np.unique(masks)
        for label_num in unique_labels:
            if label_num == 0:
                continue

            instance_mask = masks == label_num
            label_surface = np.sum(instance_mask)
            if label_surface > SURFACE_THRESHOLD_SUP or label_surface < SURFACE_THRESHOLD_INF:
                masks[instance_mask] = 0  # Remove the label

        # Store data for each label that meets the criteria
        for label in np.unique(masks):
            if label == 0:
                continue
            center = center_of_mass(masks == label)
            data.append({'Layer': z, 'Label': label, 'Center X': center[0], 'Center Y': center[1]})

        labels_all_slices.append(masks)
        df = pd.DataFrame(data)

    return reassign_labels(labels_all_slices, df)

def reassign_labels(labels_all_slices, df):
    new_labels_all_slices = []
    max_label = 0  # Pour garder une trace du dernier label utilisé
    seuil = np.mean(labels_all_slices[0].shape) * 0.025

    for z in range(len(labels_all_slices)):
        labels = labels_all_slices[z]
        new_labels = np.zeros_like(labels)
        layer_df = df[df['Layer'] == z]

        # Obtenir les dataframes pour les couches précédente et suivante, si elles existent
        prev_layer_df = df[df['Layer'] == z - 1] if z > 0 else None
        next_layer_df = df[df['Layer'] == z + 1] if z < len(labels_all_slices) - 1 else None

        for idx, row in layer_df.iterrows():
            label = row['Label']
            if label == 0:
                continue
            
            # Trouver les labels similaires dans les couches précédente et suivante
            current_center = np.array([row['Center X'], row['Center Y']])
            similar_label_found = False

            for adjacent_layer_df in [prev_layer_df, next_layer_df]:
                if adjacent_layer_df is not None:
                    adj_centers = adjacent_layer_df[['Center X', 'Center Y']].values
                    if len(adj_centers) > 0:
                        distances = cdist([current_center], adj_centers)
                        min_dist_idx = np.argmin(distances)
                        min_dist = distances[0, min_dist_idx]

                        if min_dist < seuil:
                            similar_label_found = True
                            adj_label = adjacent_layer_df.iloc[min_dist_idx]['Label']
                            new_labels[labels == label] = new_labels_all_slices[z - 1][labels_all_slices[z - 1] == adj_label][0] if adjacent_layer_df is prev_layer_df else max_label + 1
                            break
            if similar_label_found:
                max_label += 1

        new_labels_all_slices.append(new_labels)

    return reassign_labels_sequentially(np.stack(new_labels_all_slices, axis=0))

def reassign_labels_sequentially(labels):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]  # Exclure le label 0 s'il représente le fond
    
    # Créer un dictionnaire pour mapper les anciens labels vers les nouveaux
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}
    
    # Appliquer le mappage pour réassigner les labels
    new_labels = np.copy(labels)
    for old_label, new_label in label_mapping.items():
        new_labels[labels == old_label] = new_label
    
    return new_labels

def get_center_of_labels(label_image):
    unique_labels = np.unique(label_image)[1:]

    centers = {}

    for label in unique_labels:
        center = np.array(center_of_mass(label_image == label))
        centers[label] = center

    return centers

def get_normalized_center_of_labels(label_image):
    unique_labels = np.unique(label_image)[1:]
    dims = np.array(label_image.shape)

    normalized_centers = {}

    for label in unique_labels:
        center = np.array(center_of_mass(label_image == label))
        normalized_center = center / dims
        normalized_centers[label] = normalized_center

    return normalized_centers

def unify_labels(image1, image2, match_dico):
    """
    Applique les mêmes labels entre deux images segmentées en fonction du dictionnaire `match_dico`.

    Args:
    - image1: Première image segmentée avec des labels numériques.
    - image2: Deuxième image segmentée avec des labels numériques.
    - match_dico: Dictionnaire de correspondance entre les labels de `image1` et `image2`.

    Returns:
    - image1_sync: Première image synchronisée avec les nouveaux labels.
    - image2_sync: Deuxième image synchronisée avec les nouveaux labels.
    """
    # Trouver tous les labels uniques dans les deux images
    image1_sync = np.zeros_like(image1)
    image2_sync = np.zeros_like(image2)

    for label1, label2 in match_dico.items():
        new_label = label1

        image1_sync[image1 == label1] = new_label
        image2_sync[image2 == label2] = new_label

    return image1_sync, image2_sync




