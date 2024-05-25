import numpy as np
import SimpleITK as sitk
import re
import cv2
import sys
import pandas as pd
import czifile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from skimage.draw import disk
import imageio
from skimage.transform import resize

from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.ndimage import binary_dilation, binary_erosion
from stardist.models import StarDist2D, StarDist3D
from csbdeep.utils import normalize
from xml.etree import ElementTree as ET
from skimage.filters import threshold_otsu
from skimage import exposure
from scipy.ndimage import median_filter
from skimage.measure import label, regionprops

from scipy.ndimage import shift

def open_image(image_path):
    with czifile.CziFile(image_path) as czi:
        image_czi = czi.asarray()
        dic_dim = dict_shape(czi)  
        axes = czi.axes  
        metadata_xml = czi.metadata()
        metadata = ET.fromstring(metadata_xml)
        #save_metadata_to_txt(metadata_xml, '/Users/xavierdekeme/Desktop/metadata.txt')
        channels_dict = channels_dict_def(metadata)

    return image_czi, dic_dim, channels_dict, axes

def save_centers_to_csv(centers, csv_file_path):
    """
    Enregistre les centres des cercles dans un fichier CSV.
    
    Args:
    - centers (dict): Un dictionnaire contenant les centres des cercles.
                      Les clés sont les identifiants des cercles et les valeurs sont des tuples ou listes de coordonnées (x, y).
    - csv_file_path (str): Le chemin vers le fichier CSV à créer.
    """

    # Convertir le dictionnaire des centres en DataFrame pandas
    centers_df = pd.DataFrame.from_dict(centers, orient='index', columns=['Z', 'Y', 'X'])
    
    # Enregistrer le DataFrame dans un fichier CSV
    centers_df.to_csv(csv_file_path, index_label='CircleID')

def save_labels_to_csv(labels_image, csv_file_path):
    # Trouver les labels uniques dans l'image, en ignorant le fond (label 0)
    unique_labels = np.unique(labels_image[labels_image > 0])

    # Initialiser une liste pour stocker les données des labels
    labels_data = []

    for label in unique_labels:
        # Trouver les coordonnées des voxels qui appartiennent à ce label
        z_coords, y_coords, x_coords = np.where(labels_image == label)

        # Pour chaque voxel, ajouter ses coordonnées et son label à la liste
        for z, y, x in zip(z_coords, y_coords, x_coords):
            labels_data.append({"Label": label, "X": x, "Y": y, "Z": z})

    # Convertir la liste en DataFrame pandas
    labels_df = pd.DataFrame(labels_data)
    
    # Enregistrer le DataFrame dans un fichier CSV
    labels_df.to_csv(csv_file_path, index=False)

def save_vol_to_csv(df_volume, csv_file_path):
    """
    Enregistre les volumes des labels dans un fichier CSV.

    Args:
    - df_volume (DataFrame): DataFrame contenant les volumes des labels. 
                             Il doit avoir au moins deux colonnes : 'Label' et 'Volume'.
    - csv_file_path (str): Le chemin vers le fichier CSV à créer.
    """

    # Vérifier si le DataFrame contient les colonnes nécessaires
    if 'Label' not in df_volume.columns or 'Volume' not in df_volume.columns:
        raise ValueError("DataFrame must contain 'Label' and 'Volume' columns")

    # Enregistrer le DataFrame dans un fichier CSV
    df_volume.to_csv(csv_file_path, index=False)

def save_metadata_to_txt(metadata_xml, file_path):
    with open(file_path, 'w') as file:
        file.write(metadata_xml)

#return a dictionary of the czi shape dimensions:
def dict_shape(czi):
    return dict(zip([*czi.axes],czi.shape))

def channels_dict_def(metadata):
    channels = metadata.findall('.//Channel')
    channels_dict = {}
    for chan in channels:
        name = chan.attrib['Name']
        dye_element = chan.find('DyeName')
        if dye_element is not None:
            dye_numbers = re.findall(r'\d+', dye_element.text)
            dye_number = dye_numbers[-1] if dye_numbers else 'Unknown'
            channels_dict[dye_number] = name
    return channels_dict

def czi_slicer(arr,axes,indexes={"S":0, "C":0}):
    ret = arr
    axes = [*axes]
    for k,v in indexes.items():
        index = axes.index(k)
        axes.remove(k)
        ret = ret.take(axis=index, indices=v)
 
    ret_axes = ""
    for i,v in enumerate(ret.shape):
        if v > 1:
            ret_axes+=axes[i]
    return ret.squeeze(),ret_axes

def normalize_channel(channel, min_val=0, max_val=255):
    channel_normalized = exposure.rescale_intensity(channel, out_range=(min_val, max_val))
    return channel_normalized

def get_channel_index(channel_name, channel_dict):
    count = 0
    for keys, values in channel_dict.items():
        if values == channel_name:
            return count
        else:
            count += 1

def isolate_and_normalize_channel(image, dic_dim, channel_dict, TAG, axes, TAG_name):
    channel_name = channel_dict[TAG]
    channel_index = get_channel_index(channel_name, channel_dict)
    if channel_index < 0 or channel_index >= dic_dim['C']:
        raise ValueError("Channel index out of range.")

    image_czi_reduite, axes_restants = czi_slicer(image, axes, indexes={'C': channel_index})

    #channel_normalized = normalize_channel(image_czi_reduite)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/{TAG_name}.tif', image_czi_reduite.astype(np.uint16))
    return image_czi_reduite.astype(np.uint16)

def reassign_labels(labels_all_slices, df, seuil=10):
    new_labels_all_slices = []
    max_label = 0  # Pour garder une trace du dernier label utilisé

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
            
            # if not similar_label_found:
            #     new_labels[labels == label] = 0
            # else:
            #     max_label += 1

        new_labels_all_slices.append(new_labels)

    return np.stack(new_labels_all_slices, axis=0)

def reassign_labels_2layers(labels_all_slices, df, seuil=5):
    new_labels_all_slices = np.zeros_like(labels_all_slices)  # Créer un tableau de zéros de la même forme que labels_all_slices

    for z in range(len(labels_all_slices)):
        labels = labels_all_slices[z]  # Labels de la couche actuelle
        new_labels = np.zeros_like(labels)  # Initialiser new_labels pour la couche actuelle
        layer_df = df[df['Layer'] == z]

        for idx, row in layer_df.iterrows():
            label = row['Label']
            if label == 0:
                continue

            current_center = np.array([row['Center X'], row['Center Y']])
            similar_label_found = False

            # Examiner les couches -2, -1, +1 et +2
            for offset in [-2, -1, 1, 2]:
                if 0 <= z + offset < len(labels_all_slices): 
                    adjacent_layer_df = df[df['Layer'] == z + offset]

                    if not adjacent_layer_df.empty:
                        adj_centers = adjacent_layer_df[['Center X', 'Center Y']].values
                        distances = cdist([current_center], adj_centers)
                        min_dist_idx = np.argmin(distances)
                        min_dist = distances[0, min_dist_idx]

                        if min_dist < seuil:
                            similar_label_found = True
                            adj_label = adjacent_layer_df.iloc[min_dist_idx]['Label']
                            new_labels[labels == label] = adj_label  # Assigner le label trouvé dans la couche adjacente
                            break

            if not similar_label_found:
                new_labels[labels == label] = label  # Conserver le label original si aucun label similaire n'est trouvé

        new_labels_all_slices[z] = new_labels  # Assigner les nouveaux labels à la couche actuelle

    return new_labels_all_slices

def reassign_labels_new(labels_all_slices, df, seuil=10):
    new_labels_all_slices = []
    max_label = 0  # Pour garder une trace du dernier label utilisé

    for z in range(len(labels_all_slices)):
        labels = labels_all_slices[z]
        new_labels = np.zeros_like(labels)
        layer_df = df[df['Layer'] == z]

        prev_layer_df = df[df['Layer'] == z - 1] if z > 0 else None
        next_layer_df = df[df['Layer'] == z + 1] if z < len(labels_all_slices) - 1 else None

        for idx, row in layer_df.iterrows():
            label = row['Label']
            if label == 0:
                continue

            current_center = np.array([row['Center X'], row['Center Y']])
            similar_label_found = False

            # Vérifiez d'abord la couche précédente
            if prev_layer_df is not None:
                similar_label_found = check_and_assign_label(prev_layer_df, current_center, seuil, labels, label, new_labels, new_labels_all_slices, z, max_label, True)
                if similar_label_found:
                    max_label += 1

            # Si aucun label similaire n'a été trouvé dans la couche précédente, vérifiez la couche suivante
            if not similar_label_found and next_layer_df is not None:
                similar_label_found = check_and_assign_label(next_layer_df, current_center, seuil, labels, label, new_labels, new_labels_all_slices, z, max_label, False)
                if similar_label_found:
                    max_label += 1

            # Si aucun label similaire n'a été trouvé dans les couches adjacentes, ne pas réassigner ce label
            if not similar_label_found:
                new_labels[labels == label] = max_label + 1
                max_label += 1

        new_labels_all_slices.append(new_labels)

    return np.stack(new_labels_all_slices, axis=0)

def check_and_assign_label(adj_layer_df, current_center, seuil, labels, label, new_labels, new_labels_all_slices, z, max_label, is_prev_layer):
    adj_centers = adj_layer_df[['Center X', 'Center Y']].values
    if len(adj_centers) > 0:
        distances = cdist([current_center], adj_centers)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[0, min_dist_idx]

        if min_dist < seuil:
            adj_label = adj_layer_df.iloc[min_dist_idx]['Label']
            assigned_label = new_labels_all_slices[z - 1][labels_all_slices[z - 1] == adj_label][0] if is_prev_layer else max_label + 1
            new_labels[labels == label] = assigned_label
            return True
    return False

def reassign_labels_new2(labels_all_slices, df, seuil=5):
    new_labels_all_slices = np.zeros_like(labels_all_slices)

    # Projection 2D de tous les centres avec leur couche respective
    all_centers = []
    for z in range(len(labels_all_slices)):
        layer_df = df[df['Layer'] == z]
        for idx, row in layer_df.iterrows():
            center = np.array([row['Center X'], row['Center Y']])
            all_centers.append((center, z))  # Ajout du centre avec la couche

    # Association des centres proches les uns des autres
    center_to_label = {}
    label = 1  # Commencer les labels à 1

    for i, (center_a, z_a) in enumerate(all_centers):
        for j, (center_b, z_b) in enumerate(all_centers):
            if i >= j:  # Éviter de comparer deux fois le même couple de centres
                continue
            distance = np.linalg.norm(center_a - center_b)
            if distance < seuil:
                if i not in center_to_label and j not in center_to_label:
                    # Si aucun des centres n'est encore associé à un label
                    center_to_label[i] = center_to_label[j] = label
                    label += 1
                elif i in center_to_label and j not in center_to_label:
                    # Si le centre A est déjà associé mais pas le centre B
                    center_to_label[j] = center_to_label[i]
                elif j in center_to_label and i not in center_to_label:
                    # Si le centre B est déjà associé mais pas le centre A
                    center_to_label[i] = center_to_label[j]

        # Si le centre est sur la première ou la dernière couche et n'est pas encore associé, lui attribuer un nouveau label
        if (z_a == 0 or z_a == len(labels_all_slices) - 1) and i not in center_to_label:
            center_to_label[i] = label
            label += 1

    # Assigner les labels à travers les couches
    for z in range(len(labels_all_slices)):
        labels = labels_all_slices[z]
        new_labels = np.zeros_like(labels)
        layer_df = df[df['Layer'] == z]

        for idx, row in layer_df.iterrows():
            label = row['Label']
            if label == 0:  # Ignorer le fond
                continue
            center_idx = next((i for i, (center, layer) in enumerate(all_centers) if layer == z and np.array_equal(center, np.array([row['Center X'], row['Center Y']]))), None)
            if center_idx in center_to_label:
                new_labels[labels == label] = center_to_label[center_idx]

        new_labels_all_slices[z] = new_labels

    return np.stack(new_labels_all_slices, axis=0)

def find_centers_of_labels_in_3d(labels_3d):
    unique_labels = np.unique(labels_3d)
    centers = {}
    
    for label in unique_labels:
        if label == 0:  # ignore background
            continue
        # Trouver le centre de masse pour le label actuel
        center = center_of_mass(labels_3d == label)
        centers[label] = center

    return centers

def normalize_centers(labels_3d):
    unique_labels = np.unique(labels_3d)
    centers = {}
    
    # Obtenez les dimensions de l'image
    depth, height, width = labels_3d.shape
    
    for label in unique_labels:
        if label == 0:  # ignore background
            continue
        
        # Trouver le centre de masse pour le label actuel
        center = center_of_mass(labels_3d == label)
        
        # Normaliser les coordonnées du centre par rapport aux dimensions de l'image ()
        normalized_center = (center[0] / depth, center[1] / height, center[2] / width)
        
        centers[label] = normalized_center

    return centers

def plot_3d_centers(centers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Séparez les coordonnées z, y, x
    zs = [center[0] for center in centers.values()]
    ys = [center[1] for center in centers.values()]
    xs = [center[2] for center in centers.values()]

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    plt.show()

def plot_3d_centers_all(centers_TAG1, centers_TAG2, common_centers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Tracer les centres de TAG1
    zs_TAG1 = [center[0] for center in centers_TAG1.values()]
    ys_TAG1 = [center[1] for center in centers_TAG1.values()]
    xs_TAG1 = [center[2] for center in centers_TAG1.values()]
    ax.scatter(xs_TAG1, ys_TAG1, zs_TAG1, color='r', label='TAG1')

    # Tracer les centres de TAG2
    zs_TAG2 = [center[0] for center in centers_TAG2.values()]
    ys_TAG2 = [center[1] for center in centers_TAG2.values()]
    xs_TAG2 = [center[2] for center in centers_TAG2.values()]
    ax.scatter(xs_TAG2, ys_TAG2, zs_TAG2, color='b', label='TAG2')

    # Tracer les centres communs
    zs_common = [center[0] for center in common_centers.values()]
    ys_common = [center[1] for center in common_centers.values()]
    xs_common = [center[2] for center in common_centers.values()]
    ax.scatter(xs_common, ys_common, zs_common, color='g', label='Commun')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()

    plt.show()

def plot_label_areas(labels_3d):
    depth = labels_3d.shape[0]
    
    rows = int(np.ceil(np.sqrt(depth)))
    cols = int(np.ceil(depth / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))  # Ajustez la taille de la figure selon vos besoins
    axes_flat = axes.flatten()

    for z in range(depth):
        layer = labels_3d[z, :, :]
        unique_labels, counts = np.unique(layer, return_counts=True)
        
        valid_indices = unique_labels != 0
        unique_labels = unique_labels[valid_indices]
        counts = counts[valid_indices]

        # Tri des labels et des aires
        sorted_areas_indices = np.argsort(counts)
        sorted_labels = unique_labels[sorted_areas_indices]
        sorted_areas = counts[sorted_areas_indices]

        # Création du graphique en bâtonnets pour chaque couche avec les labels triés
        axes_flat[z].bar(range(len(sorted_labels)), sorted_areas, color='skyblue', tick_label=sorted_labels)
        axes_flat[z].set_title(f'Layer {z + 1}')
        axes_flat[z].set_xlabel('Sorted Label Number')
        axes_flat[z].set_ylabel('Area (pixels)')

        # Rotation des étiquettes sur l'axe x pour une meilleure lisibilité
        for label in axes_flat[z].get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    
    # Masquer les axes inutilisés
    for ax in axes_flat[depth:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def do_segmentation(model, isolate_image, SURFACE_THRESHOLD_SUP, SURFACE_THRESHOLD_INF):
    labels_all_slices = []
    data = []

    for z in range(isolate_image.shape[0]):
        img_slice = isolate_image[z, :, :]
        img_slice = normalize(img_slice, 1, 99.8)
        filtered_img_slice = median_filter(img_slice, size=3)
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

    return labels_all_slices, data

def update_labels_image(labels_image, labels_to_keep):
    updated_labels_image = np.zeros_like(labels_image)
    for label in labels_to_keep:
        updated_labels_image[labels_image == label] = label
    return updated_labels_image

def keep_regions_with_all_tags(centers_TAG1, centers_TAG2, centers_TAG3, distance_threshold=0.1, weights=(1, 1, 1)):
    updated_centers_TAG1 = {}
    updated_centers_TAG2 = {}
    updated_centers_TAG3 = {}

    # Appliquer les poids aux coordonnées des centres
    def apply_weights(coordinates, weights):
        return np.multiply(coordinates, weights)

    coordinates_TAG1_weighted = apply_weights(np.array(list(centers_TAG1.values())), weights)
    coordinates_TAG2_weighted = apply_weights(np.array(list(centers_TAG2.values())), weights)
    coordinates_TAG3_weighted = apply_weights(np.array(list(centers_TAG3.values())), weights)

    for tag_dict, tag_coords_weighted, other_coords1_weighted, other_coords2_weighted, updated_dict in [
        (centers_TAG1, coordinates_TAG1_weighted, coordinates_TAG2_weighted, coordinates_TAG3_weighted, updated_centers_TAG1),
        (centers_TAG2, coordinates_TAG2_weighted, coordinates_TAG1_weighted, coordinates_TAG3_weighted, updated_centers_TAG2),
        (centers_TAG3, coordinates_TAG3_weighted, coordinates_TAG1_weighted, coordinates_TAG2_weighted, updated_centers_TAG3)]:

        for label, point in tag_dict.items():
            point_weighted = np.multiply(point, weights)

            distances_to_other1 = cdist([point_weighted], other_coords1_weighted)
            distances_to_other2 = cdist([point_weighted], other_coords2_weighted)

            nearest_other1_idx = np.argmin(distances_to_other1) if np.any(distances_to_other1 <= distance_threshold) else None
            nearest_other2_idx = np.argmin(distances_to_other2) if np.any(distances_to_other2 <= distance_threshold) else None

            if nearest_other1_idx is not None and nearest_other2_idx is not None:
                updated_dict[label] = point
                if tag_dict is centers_TAG1:
                    updated_centers_TAG2[list(centers_TAG2.keys())[nearest_other1_idx]] = other_coords1_weighted[nearest_other1_idx] / weights
                    updated_centers_TAG3[list(centers_TAG3.keys())[nearest_other2_idx]] = other_coords2_weighted[nearest_other2_idx] / weights
                elif tag_dict is centers_TAG2:
                    updated_centers_TAG1[list(centers_TAG1.keys())[nearest_other1_idx]] = other_coords1_weighted[nearest_other1_idx] / weights
                    updated_centers_TAG3[list(centers_TAG3.keys())[nearest_other2_idx]] = other_coords2_weighted[nearest_other2_idx] / weights
                else:  # tag_dict is centers_TAG3
                    updated_centers_TAG1[list(centers_TAG1.keys())[nearest_other1_idx]] = other_coords1_weighted[nearest_other1_idx] / weights
                    updated_centers_TAG2[list(centers_TAG2.keys())[nearest_other2_idx]] = other_coords2_weighted[nearest_other2_idx] / weights

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3

def calculate_shift(pairs):
    """
    Calcule le déplacement moyen à partir des paires de points.

    Args:
    - pairs (list): Liste de tuples contenant les paires de points (point_img1, point_img2).

    Returns:
    - shift_values (tuple): Le déplacement moyen en x et y.
    """
    shifts = [np.array(point_img2) - np.array(point_img1) for point_img1, point_img2 in pairs]
    shift_values = np.mean(shifts, axis=0)
    return shift_values

def shift_image(image, shift_values):
    """
    Déplace l'image en fonction des valeurs de déplacement.

    Args:
    - image (ndarray): Image à déplacer.
    - shift_values (tuple): Valeurs de déplacement en x et y.

    Returns:
    - shifted_image (ndarray): Image déplacée.
    """
    shifted_image = np.zeros_like(image)

    # Application du déplacement à chaque couche z de l'image
    for z in range(image.shape[0]):
        shifted_image[z, :, :] = shift(image[z, :, :], shift_values)

    return shifted_image

def adjust_labels_and_calculate_volumes(labels_all_slices):
    adjusted_labels_all_slices = np.copy(labels_all_slices)
    labels_area = {}

    for z in range(labels_all_slices.shape[0]):
        for label in np.unique(labels_all_slices[z, :, :]):
            if label == 0:
                continue
            
            current_label_mask = labels_all_slices[z] == label
            current_label_area = np.sum(current_label_mask)

            if f"Label {label}" not in labels_area:
                labels_area[f"Label {label}"] = {z: current_label_area}
            else:
                labels_area[f"Label {label}"][z] = current_label_area

    volumes = {label: sum(areas.values()) for label, areas in labels_area.items()}
    df_volumes = pd.DataFrame(list(volumes.items()), columns=['Label', 'Volume'])
    
    for label, layers in labels_area.items():
        layer_numbers = sorted(layers.keys())
        for i, z in enumerate(layer_numbers):
            current_area = layers[z]
            if i > 0 and i < len(layer_numbers)-1:
                prev_area = layers[layer_numbers[i - 1]]
                next_area = layers[layer_numbers[i + 1]]
                target_area = (prev_area + next_area) / 2 

                if abs(current_area - target_area) > current_area * 0.25:
                    current_label_mask = labels_all_slices[z] == int(label.split(" ")[-1])
                    # Supprimer la segmentation problématique en réinitialisant cette région dans l'image des labels ajustés
                    adjusted_labels_all_slices[z][current_label_mask] = 0

                    # Remplacer par une segmentation "normale"
                    # Cette étape dépend de ce que vous avez disponible ou de ce que vous considérez comme une segmentation "normale"
                    # Par exemple, utiliser la segmentation de la couche précédente si z > 0
                    if z > 0:
                        # Utiliser la segmentation de la couche précédente
                        previous_label_mask = labels_all_slices[z - 1] == int(label.split(" ")[-1])
                        adjusted_labels_all_slices[z][previous_label_mask] = int(label.split(" ")[-1])
                    elif z < len(labels_all_slices) - 1:
                        # Ou utiliser la segmentation de la couche suivante si z est la première couche
                        next_label_mask = labels_all_slices[z + 1] == int(label.split(" ")[-1])
                        adjusted_labels_all_slices[z][next_label_mask] = int(label.split(" ")[-1])

    return adjusted_labels_all_slices, df_volumes

def filter_labels_by_shape_3d(labels_all_slices, seuil_area=50, seuil_eccentricity=0.975):
    # Détecter les labels non conformes dans chaque tranche
    non_conforming_labels = set()

    # Itérer sur chaque tranche de l'image 3D
    for z in range(labels_all_slices.shape[0]):
        slice_labels = labels_all_slices[z, :, :]

        # Calculer les propriétés pour chaque label dans la tranche
        for region in regionprops(slice_labels):
            # Utiliser l'aire et l'excentricité comme critères pour les formes circulaires/ovales
            if region.area < seuil_area or region.eccentricity > seuil_eccentricity:
                # Ajouter le label à la liste des labels non conformes
                non_conforming_labels.add(region.label)

    # Itérer de nouveau sur chaque tranche pour retirer les labels non conformes
    for z in range(labels_all_slices.shape[0]):
        for non_conforming_label in non_conforming_labels:
            labels_all_slices[z, :, :][labels_all_slices[z, :, :] == non_conforming_label] = 0

    return labels_all_slices



IMAGE_NAME = '1.IF1'
IMAGE_NAME_2 = '1.IF2'
TAG1 = '647'
TAG2 = '546'
TAG3 = '405'
TAG4 = '405'
TAG5 = '594' #CHECK AVEC SARAH (pas le meme que slide)
TAG6 = '633'
TAG7 = '488' 
PHALLOIDINE = '488'
image_path = f'/Volumes/LaCie/Mémoire/Images/New/{IMAGE_NAME}.czi'
image_czi_1, dic_dim_1, channels_dict_1, axes_1 = open_image(image_path)
image_path_2 = f'/Volumes/LaCie/Mémoire/Images/New/{IMAGE_NAME_2}.czi'
image_czi_2, dic_dim_2, channels_dict_2, axes_2 = open_image(image_path_2)


TAG1_img = isolate_and_normalize_channel(image_czi_1, dic_dim_1, channels_dict_1, TAG1, axes_1, "TAG1")
TAG2_img = isolate_and_normalize_channel(image_czi_1, dic_dim_1, channels_dict_1, TAG2, axes_1, "TAG2")
TAG3_img = isolate_and_normalize_channel(image_czi_1, dic_dim_1, channels_dict_1, TAG3, axes_1, "TAG3")
TAG4_img = isolate_and_normalize_channel(image_czi_2, dic_dim_2, channels_dict_2, TAG4, axes_2, "TAG4")
TAG5_img = isolate_and_normalize_channel(image_czi_2, dic_dim_2, channels_dict_2, TAG5, axes_2, "TAG5")
TAG6_img = isolate_and_normalize_channel(image_czi_2, dic_dim_2, channels_dict_2, TAG6, axes_2, "TAG6")
TAG7_img = isolate_and_normalize_channel(image_czi_2, dic_dim_2, channels_dict_2, TAG7, axes_2, "TAG7")
TAG1_img_cropped = TAG1_img[:, int(1024/2):, int(1024/2):]
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG1_cropped.tif', TAG1_img_cropped)
TAG2_img_cropped = TAG2_img[:, int(1024/2):, int(1024/2):]
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG2_cropped.tif', TAG2_img_cropped)
TAG3_img_cropped = TAG3_img[:, int(1024/2):, int(1024/2):]
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG3_cropped.tif', TAG3_img_cropped)

TAG4_img_trimmed = TAG4_img[6:-6, :, :]
TAG4_img_cropped = TAG4_img[6:-6, int(1024/2):, int(1024/2):]
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG4_trimmed.tif', TAG4_img_trimmed)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG4_cropped.tif', TAG4_img_cropped)
TAG5_img_trimmed = TAG5_img[6:-6, :, :]
TAG5_img_cropped = TAG5_img[6:-6, int(1024/2):, int(1024/2):]
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG5_trimmed.tif', TAG5_img_trimmed)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG5_cropped.tif', TAG5_img_cropped)
TAG6_img_trimmed = TAG6_img[6:-6, :, :]
TAG6_img_cropped = TAG6_img[6:-6, int(1024/2):, int(1024/2):]
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG6_trimmed.tif', TAG6_img_trimmed)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG6_cropped.tif', TAG6_img_cropped)
TAG7_img_trimmed = TAG7_img[6:-6, :, :]
TAG7_img_cropped = TAG7_img[6:-6, int(1024/2):, int(1024/2):]
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG7_trimmed.tif', TAG7_img_trimmed)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/TAG7_cropped.tif', TAG7_img_cropped)



PHALLOIDINE_img = isolate_and_normalize_channel(image_czi_1, dic_dim_1, channels_dict_1, PHALLOIDINE, axes_1, "PHALLOIDINE")
PHALLOIDINE_img_cropped = PHALLOIDINE_img[:, int(1024/2):, int(1024/2):]
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/PHALLOIDINE_cropped.tif', PHALLOIDINE_img_cropped)


images1 = [TAG1_img, TAG2_img, TAG3_img]
images1_cropped = [TAG1_img_cropped, TAG2_img_cropped, TAG3_img_cropped]
images1_cropped_phalo = [TAG1_img_cropped, TAG2_img_cropped, TAG3_img_cropped, PHALLOIDINE_img_cropped]
images1_phalo = [TAG1_img, TAG2_img, TAG3_img, PHALLOIDINE_img]
images2 = [TAG4_img_trimmed, TAG5_img_trimmed, TAG6_img_trimmed, TAG7_img_trimmed]
images2_cropped = [TAG4_img_cropped, TAG5_img_cropped, TAG6_img_cropped, TAG7_img_cropped]

composite_image = np.maximum.reduce(images1)
composite_image_cropped = np.maximum.reduce(images1_cropped)
composite_image_cropped_phalo = np.maximum.reduce(images1_cropped_phalo)
composite_image_bis = np.maximum.reduce(images1_phalo)
composite_image2 = np.maximum.reduce(images2)
composite_image2_cropped = np.maximum.reduce(images2_cropped)


imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG123.tif', composite_image)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG123_cropped.tif', composite_image_cropped)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG123_cropped_phalo.tif', composite_image_cropped_phalo)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG123_phalo.tif', composite_image_bis)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG4567.tif', composite_image2)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG4567_cropped.tif', composite_image2_cropped)


# Charger le modèle pré-entraîné StarDist 2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# Initialisation d'un dictionnaire pour stocker les informations par TAG
tags_data = {}

labels_all_slices, data = do_segmentation(model, composite_image_cropped, 650, 50)
labels_all_slices = np.stack(labels_all_slices, axis=0)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_full_TAG123_cropped.tif', labels_all_slices)
df = pd.DataFrame(data)

labels = reassign_labels_new2(labels_all_slices, df)
adjusted_labels, df_volumes = adjust_labels_and_calculate_volumes(labels)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_fused_TAG123_cropped.tif', labels)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_fused_TAG123_cropped.tif', adjusted_labels)


labels_all_slices, data = do_segmentation(model, composite_image2_cropped, 700, 50)
labels_all_slices = np.stack(labels_all_slices, axis=0)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_full_TAG4567_cropped.tif', labels_all_slices)
df = pd.DataFrame(data)

labels = reassign_labels_new2(labels_all_slices, df)
adjusted_labels, df_volumes = adjust_labels_and_calculate_volumes(labels)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_fused_TAG4567_cropped.tif', labels)
imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_fused_TAG4567_cropped.tif', adjusted_labels)



# Liste des informations des TAGs
tags_info = [
    {"tag_img": TAG1_img_cropped, "thresholds": (700, 75), "tag_name": "TAG1"},
    {"tag_img": TAG2_img_cropped, "thresholds": (700, 75), "tag_name": "TAG2"},
    {"tag_img": TAG3_img_cropped, "thresholds": (700, 75), "tag_name": "TAG3"},
]

tags_info_2 = [
    {"tag_img": TAG4_img_cropped, "thresholds": (700, 75), "tag_name": "TAG4"},
    {"tag_img": TAG5_img_cropped, "thresholds": (700, 75), "tag_name": "TAG5"},
    {"tag_img": TAG6_img_cropped, "thresholds": (700, 75), "tag_name": "TAG6"},
    {"tag_img": TAG7_img_cropped, "thresholds": (700, 75), "tag_name": "TAG7"},
]

tags_info_full = [
    {"tag_img": composite_image_cropped, "thresholds": (10000, 500), "tag_name": "TAG123_full"},
]

tags_info_2_full = [
    {"tag_img": composite_image2_cropped, "thresholds": (10000, 500), "tag_name": "TAG4567_full"},
]

for tag_info in tags_info_full:
    labels_all_slices, data = do_segmentation(model, tag_info["tag_img"], *tag_info["thresholds"])
    labels_all_slices = np.stack(labels_all_slices, axis=0)
    df = pd.DataFrame(data)

    labels = reassign_labels(labels_all_slices, df)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_full_{tag_info["tag_name"]}.tif', labels)

    centers = find_centers_of_labels_in_3d(labels)
    centers_norm = normalize_centers(labels)

for tag_info in tags_info_2_full:
    labels_all_slices, data = do_segmentation(model, tag_info["tag_img"], *tag_info["thresholds"])
    labels_all_slices = np.stack(labels_all_slices, axis=0)
    df = pd.DataFrame(data)

    print(tag_info["tag_name"])
    labels = reassign_labels(labels_all_slices, df)

    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_full_{tag_info["tag_name"]}.tif', labels)

    centers = find_centers_of_labels_in_3d(labels)
    centers_norm = normalize_centers(labels)



for tag_info in tags_info:
    labels_all_slices, data = do_segmentation(model, tag_info["tag_img"], *tag_info["thresholds"])
    labels_all_slices = np.stack(labels_all_slices, axis=0)
    df = pd.DataFrame(data)

    labels = reassign_labels(labels_all_slices, df)
    adjusted_labels, df_volumes = adjust_labels_and_calculate_volumes(labels)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_{tag_info["tag_name"]}_cropped.tif', labels)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_{tag_info["tag_name"]}_cropped.tif', adjusted_labels)

    centers = find_centers_of_labels_in_3d(adjusted_labels)
    centers_norm = normalize_centers(adjusted_labels)

    tags_data[tag_info["tag_name"]] = {
        "labels": labels,
        "centers": centers,
        "centers_norm": centers_norm,
        "df_volume": df_volumes,
        "dataframe": df,
    }


for tag_info in tags_info_2:
    labels_all_slices, data = do_segmentation(model, tag_info["tag_img"], *tag_info["thresholds"])
    labels_all_slices = np.stack(labels_all_slices, axis=0)
    df = pd.DataFrame(data)

    print(tag_info["tag_name"])
    labels = reassign_labels(labels_all_slices, df)
    adjusted_labels, df_volumes = adjust_labels_and_calculate_volumes(labels)

    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_{tag_info["tag_name"]}_cropped.tif', labels)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_{tag_info["tag_name"]}_cropped.tif', adjusted_labels)

    centers = find_centers_of_labels_in_3d(adjusted_labels)
    centers_norm = normalize_centers(adjusted_labels)

    tags_data[tag_info["tag_name"]] = {
        "labels": labels,
        "centers": centers,
        "centers_norm": centers_norm,
        "df_volume": df_volumes,
        "dataframe": df,
    }

tag1_centers_norm = tags_data["TAG1"]["centers_norm"]
tag2_centers_norm = tags_data["TAG2"]["centers_norm"]
tag3_centers_norm = tags_data["TAG3"]["centers_norm"]
tag4_centers_norm = tags_data["TAG4"]["centers_norm"]
tag5_centers_norm = tags_data["TAG5"]["centers_norm"]
tag6_centers_norm = tags_data["TAG6"]["centers_norm"]
tag7_centers_norm = tags_data["TAG7"]["centers_norm"]

tag1_labels = tags_data["TAG1"]["labels"]
tag2_labels = tags_data["TAG2"]["labels"]
tag3_labels = tags_data["TAG3"]["labels"]
tag4_labels = tags_data["TAG4"]["labels"]
tag5_labels = tags_data["TAG5"]["labels"]
tag6_labels = tags_data["TAG6"]["labels"]
tag7_labels = tags_data["TAG7"]["labels"]

tag1_vol = tags_data["TAG1"]["df_volume"]
tag2_vol = tags_data["TAG2"]["df_volume"]
tag3_vol = tags_data["TAG3"]["df_volume"]
tag4_vol = tags_data["TAG4"]["df_volume"]
tag5_vol = tags_data["TAG5"]["df_volume"]
tag6_vol = tags_data["TAG6"]["df_volume"]
tag7_vol = tags_data["TAG7"]["df_volume"]

index = [1, 2, 3, 4, 5, 6, 7]
for i in index:
    center_norm_file_path = f"/Users/xavierdekeme/Desktop/Data/CentreNorm/centers{i}.csv"
    label_file_path = f"/Users/xavierdekeme/Desktop/Data/Label/label{i}.csv"
    volume_file_path = f"/Users/xavierdekeme/Desktop/Data/Volume/volume{i}.csv"
    
    center_norm_data = tags_data[f"TAG{i}"]["centers_norm"]
    label_data = tags_data[f"TAG{i}"]["labels"]
    vol_data = tags_data[f"TAG{i}"]["df_volume"]

    save_centers_to_csv(center_norm_data, center_norm_file_path)
    save_labels_to_csv(label_data, label_file_path)
    save_vol_to_csv(vol_data, volume_file_path)
    












# #FIND POPULATION
# update_center1, update_center2, update_center7  = keep_regions_with_all_tags(tag1_centers_norm, tag2_centers_norm, tag7_centers_norm, distance_threshold=0.075, weights=(1, 1, 0.25))

# print(update_center1)
# print(update_center2)
# print(update_center7)

# updated_labels_images = update_labels_image(tags_data["TAG1"]["labels"], update_center1)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_TAG1_update_pop127.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tags_data["TAG2"]["labels"], update_center2)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_TAG2_update_pop127.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tags_data["TAG7"]["labels"], update_center7)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_TAG7_update_pop127.tif', updated_labels_images)



# update_center1, update_center2, update_center3  = keep_regions_with_all_tags(tag1_centers_norm, tag2_centers_norm, tag3_centers_norm, distance_threshold=0.075, weights=(1, 1, 0.25))

# print(update_center1)
# print(update_center2)
# print(update_center3)

# updated_labels_images = update_labels_image(tags_data["TAG1"]["labels"], update_center1)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_TAG1_update_pop123.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tags_data["TAG2"]["labels"], update_center2)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_TAG2_update_pop123.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tags_data["TAG3"]["labels"], update_center3)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_TAG3_update_pop123.tif', updated_labels_images)


# update_center1, update_center5, update_center7  = keep_regions_with_all_tags(tag1_centers_norm, tag5_centers_norm, tag7_centers_norm, distance_threshold=0.075, weights=(1, 1, 0.25))

# print(update_center1)
# print(update_center5)
# print(update_center7)

# updated_labels_images = update_labels_image(tags_data["TAG1"]["labels"], update_center1)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_TAG1_update_pop157.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tags_data["TAG5"]["labels"], update_center5)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_TAG5_update_pop157.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tags_data["TAG7"]["labels"], update_center7)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_TAG7_update_pop157.tif', updated_labels_images)



