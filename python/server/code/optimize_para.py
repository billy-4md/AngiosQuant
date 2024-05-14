import numpy as np
import re
import cv2
import sys
import pandas as pd
import czifile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import open3d as o3d

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.spatial import KDTree
from skimage.draw import disk
import imageio
from skimage.transform import resize

from scipy.ndimage import gaussian_filter, center_of_mass
from stardist.models import StarDist2D, StarDist3D
from csbdeep.utils import normalize
from xml.etree import ElementTree as ET
from skimage.filters import threshold_otsu
from skimage import exposure

from sklearn.decomposition import PCA
from scipy.ndimage import affine_transform
from scipy.signal import correlate2d
import scipy.ndimage
from simpleicp import PointCloud, SimpleICP
from PIL import Image


def read_centers_from_csv(csv_file_path):
    """
    Lit les centres des cercles à partir d'un fichier CSV et les convertit en dictionnaire.
    
    Args:
    - csv_file_path (str): Le chemin vers le fichier CSV à lire.
    
    Returns:
    - Un dictionnaire contenant les centres des cercles. Les clés sont les identifiants des cercles et les valeurs sont des tuples de coordonnées (x, y, z).
    """
    centers_df = pd.read_csv(csv_file_path)
    centers_dict = {row.CircleID: (row.Z, row.Y, row.X) for index, row in centers_df.iterrows()}
    return centers_dict

def read_labels_from_csv(csv_file_path, image_shape):
    """
    Lit les labels à partir d'un fichier CSV et les reconstruit dans une image de labels (tableau NumPy).
    
    Args:
    - csv_file_path (str): Le chemin vers le fichier CSV à lire.
    - image_shape (tuple): La forme de l'image de labels à reconstruire (z, y, x).
    
    Returns:
    - Un tableau NumPy représentant l'image de labels reconstruite.
    """
    labels_df = pd.read_csv(csv_file_path)
    labels_image = np.zeros(image_shape, dtype=int)
    for index, row in labels_df.iterrows():
        labels_image[row.Z, row.Y, row.X] = row.Label
    return labels_image

def read_labels_from_csv_without_2(csv_file_path, image_shape):
    """
    Lit les labels à partir d'un fichier CSV, exclut ceux présents sur deux couches ou moins,
    et reconstruit l'image de labels dans un tableau NumPy.

    Args:
    - csv_file_path (str): Le chemin vers le fichier CSV à lire.
    - image_shape (tuple): La forme de l'image de labels à reconstruire (z, y, x).

    Returns:
    - Un tableau NumPy représentant l'image de labels reconstruite.
    """
    labels_df = pd.read_csv(csv_file_path)
    labels_image = np.zeros(image_shape, dtype=int)

    # Compter le nombre de couches sur lesquelles chaque label apparaît
    layer_counts = labels_df.groupby('Label')['Z'].nunique()

    # Filtrer les labels qui apparaissent sur plus de deux couches
    valid_labels = layer_counts[layer_counts > 2].index

    # Reconstruire l'image de labels en utilisant uniquement les labels valides
    for index, row in labels_df.iterrows():
        if row.Label in valid_labels:
            labels_image[row.Z, row.Y, row.X] = row.Label

    return labels_image

def read_volumes_from_csv(csv_file_path):
    """
    Lit les volumes des labels à partir d'un fichier CSV, les normalise en divisant chaque volume par le volume maximum,
    et les convertit en dictionnaire.
    
    Args:
    - csv_file_path (str): Le chemin vers le fichier CSV à lire.
    
    Returns:
    - Un dictionnaire contenant les volumes normalisés des labels. Les clés sont les identifiants des labels et les valeurs sont les volumes normalisés.
    """
    volumes_df = pd.read_csv(csv_file_path)
    
    # Trouver le volume maximum dans les données
    max_volume = volumes_df['Volume'].max()
    
    # Vérifier que max_volume n'est pas nul pour éviter la division par zéro
    if max_volume == 0:
        raise ValueError("Maximum volume is 0, cannot normalize volumes.")
    
    # Normaliser les volumes en divisant chaque volume par le volume maximum
    volumes_dict = {row['Label']: row['Volume'] for index, row in volumes_df.iterrows()}
    
    return volumes_dict

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



#vol[f"Label {int(label)}"]
def keep_matching_regions_across_tags(centers_TAG1, centers_TAG2, centers_TAG3, vol_TAG1, vol_TAG2, vol_TAG3, distance_threshold=0.1, volume_threshold=0.5, weights=(1, 1, 0.8)):
    updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3 = {}, {}, {}
    unified_label_counter = 1

    def apply_weights(coordinates, weights):
        return np.multiply(coordinates, weights)

    def check_and_update_matches(centers, other_centers1, other_centers2, vol, other_vol1, other_vol2, updated_centers, updated_other1, updated_other2):
        nonlocal unified_label_counter  # Pour accéder à la variable extérieure unified_label_counter
        for label, point in centers.items():
            point_weighted = apply_weights(point, weights)
            distances_to_other1 = cdist([point_weighted], other_centers1)
            distances_to_other2 = cdist([point_weighted], other_centers2)

            for idx1, dist1 in enumerate(distances_to_other1[0]):
                if dist1 > distance_threshold:
                    continue
                for idx2, dist2 in enumerate(distances_to_other2[0]):
                    if dist2 > distance_threshold:
                        continue

                    # Récupération des volumes et vérification des seuils
                    vol1 = vol[f"Label {int(label)}"]
                    vol2 = other_vol1[list(other_vol1.keys())[idx1]]
                    vol3 = other_vol2[list(other_vol2.keys())[idx2]]

                    if abs(vol1 - vol2) <= (volume_threshold * vol1) and abs(vol1 - vol3) <= (volume_threshold * vol1) and abs(vol2 - vol3) <= (volume_threshold * vol2):
                        unified_label = f"Unified {unified_label_counter}"
                        unified_label_counter += 1

                        updated_centers[f"Label {int(label)}"] = {'point': point, 'unified_label': unified_label}
                        updated_other1[list(other_vol1.keys())[idx1]] = {'point': other_centers1[idx1], 'unified_label': unified_label}
                        updated_other2[list(other_vol2.keys())[idx2]] = {'point': other_centers2[idx2], 'unified_label': unified_label}
                        break  # Correspondance valide trouvée, pas besoin de chercher plus loin

    coordinates_TAG1_weighted = apply_weights(np.array(list(centers_TAG1.values())), weights)
    coordinates_TAG2_weighted = apply_weights(np.array(list(centers_TAG2.values())), weights)
    coordinates_TAG3_weighted = apply_weights(np.array(list(centers_TAG3.values())), weights)

    # Vérifier les correspondances pour chaque TAG
    check_and_update_matches(centers_TAG1, coordinates_TAG2_weighted, coordinates_TAG3_weighted, vol_TAG1, vol_TAG2, vol_TAG3, updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3)
    #check_and_update_matches(centers_TAG2, coordinates_TAG1_weighted, coordinates_TAG3_weighted, vol_TAG2, vol_TAG1, vol_TAG3, updated_centers_TAG2, updated_centers_TAG1, updated_centers_TAG3)
    #check_and_update_matches(centers_TAG3, coordinates_TAG1_weighted, coordinates_TAG2_weighted, vol_TAG3, vol_TAG1, vol_TAG2, updated_centers_TAG3, updated_centers_TAG1, updated_centers_TAG2)

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3

#1-2 same images
def keep_regions_with_all_tags_NEW(centers_TAG1, centers_TAG2, centers_TAG3, distance_threshold_similar=0.05, distance_threshold_different=0.1):
    # Initialiser les dictionnaires pour stocker les centres mis à jour
    updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3 = {}, {}, {}

    # Choisir un des TAGs comme référence (ici, TAG1)
    for label1, center1 in centers_TAG1.items():
        # Créer une sphère autour du point de référence avec deux seuils de distance différents
        distances_to_TAG2 = cdist([center1], list(centers_TAG2.values()))
        distances_to_TAG3 = cdist([center1], list(centers_TAG3.values()))

        # Utiliser distance_threshold_similar pour les images similaires (ici entre TAG1 et TAG2) et distance_threshold_different pour la deuxième image (ici TAG3)
        if np.any(distances_to_TAG2 <= distance_threshold_similar) and np.any(distances_to_TAG3 <= distance_threshold_different):
            # Garder le point de référence
            updated_centers_TAG1[label1] = center1
            
            # Garder le point le plus proche de chaque autre TAG
            nearest_TAG2_idx = np.argmin(distances_to_TAG2)
            nearest_TAG3_idx = np.argmin(distances_to_TAG3)
            
            nearest_TAG2_label = list(centers_TAG2.keys())[nearest_TAG2_idx]
            nearest_TAG3_label = list(centers_TAG3.keys())[nearest_TAG3_idx]
            
            updated_centers_TAG2[nearest_TAG2_label] = centers_TAG2[nearest_TAG2_label]
            updated_centers_TAG3[nearest_TAG3_label] = centers_TAG3[nearest_TAG3_label]

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3

def keep_regions_with_best_matches(centers_TAG1, centers_TAG2, centers_TAG3, distance_threshold=0.25, top_matches=7):
    # Liste pour stocker les meilleurs matchs
    best_matches = []

    # Parcourir tous les points de TAG1
    for label1, center1 in centers_TAG1.items():
        # Calculer les distances entre le point actuel de TAG1 et tous les points de TAG2 et TAG3
        distances_to_TAG2 = cdist([center1], list(centers_TAG2.values()))
        distances_to_TAG3 = cdist([center1], list(centers_TAG3.values()))

        # Filtrer les points dans la sphère de distance_threshold pour TAG2 et TAG3
        within_threshold_TAG2 = np.where(distances_to_TAG2 <= distance_threshold)[1]  # indices des points de TAG2 dans la sphère
        within_threshold_TAG3 = np.where(distances_to_TAG3 <= distance_threshold)[1]  # indices des points de TAG3 dans la sphère

        # Parcourir tous les points filtrés de TAG2 et TAG3 pour trouver les meilleurs matchs
        for idx2 in within_threshold_TAG2:
            for idx3 in within_threshold_TAG3:
                total_distance = distances_to_TAG2[0, idx2] + distances_to_TAG3[0, idx3]
                best_matches.append((total_distance, label1, list(centers_TAG2.keys())[idx2], list(centers_TAG3.keys())[idx3]))

    # Trier les matchs par distance totale (ascendante) et garder les X meilleurs
    best_matches.sort(key=lambda x: x[0])
    top_best_matches = best_matches[:top_matches]

    # Extraire les identifiants des centres des meilleurs matchs
    updated_centers_TAG1 = {match[1]: centers_TAG1[match[1]] for match in top_best_matches}
    updated_centers_TAG2 = {match[2]: centers_TAG2[match[2]] for match in top_best_matches}
    updated_centers_TAG3 = {match[3]: centers_TAG3[match[3]] for match in top_best_matches}

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3

def keep_matching_regions_across_tags_PCA(centers_TAG1, centers_TAG2, centers_TAG3, vectors_TAG1, vectors_TAG2, vectors_TAG3, distance_threshold=0.075, angle_threshold_prim=8, angle_threshold_sec=10, weights=(0.1, 1, 1)):
    updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3 = {}, {}, {}
    unified_label_counter = 1

    def apply_weights(coordinates, weights):
        return np.multiply(coordinates, weights)

    def angle_between_vectors(v1, v2):
        """Calculate the angle between two vectors."""
        dot_product = np.dot(v1, v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return angle

    def validate_matches(centers1, centers2, centers3, vectors1, vectors2, vectors3, distance_threshold, angle_threshold_prim, angle_threshold_sec, weights):
        nonlocal unified_label_counter
        for label1, vec1 in vectors1.items():
            for label2, vec2 in vectors2.items():
                for label3, vec3 in vectors3.items():
                    vec1_prim, vec1_sec = vec1[0], vec1[1]
                    vec2_prim, vec2_sec = vec2[0], vec2[1]
                    vec3_prim, vec3_sec = vec3[0], vec3[1]
                    angle12_prim, angle12_sec = np.degrees(angle_between_vectors(vec1_prim, vec2_prim)), np.degrees(angle_between_vectors(vec1_sec, vec2_sec))
                    angle13_prim, angle13_sec  = np.degrees(angle_between_vectors(vec1_prim, vec3_prim)), np.degrees(angle_between_vectors(vec1_sec, vec3_sec))
                    angle23_prim, angle23_sec  = np.degrees(angle_between_vectors(vec2_prim, vec3_prim)), np.degrees(angle_between_vectors(vec2_sec, vec3_sec))

                    if (
                            (angle12_prim < angle_threshold_prim or abs(angle12_prim - 180) < angle_threshold_prim)
                        and (angle13_prim < angle_threshold_prim or abs(angle13_prim - 180) < angle_threshold_prim)
                        and (angle23_prim < angle_threshold_prim or abs(angle23_prim - 180) < angle_threshold_prim)
                        and (angle12_sec < angle_threshold_sec or abs(angle12_sec - 180) < angle_threshold_sec)
                        and (angle13_sec < angle_threshold_sec or abs(angle13_sec - 180) < angle_threshold_sec)
                        and (angle23_sec < angle_threshold_sec or abs(angle23_sec - 180) < angle_threshold_sec)
                    ):
                        point1 = centers1[label1]
                        point2 = centers2[label2]
                        point3 = centers3[label3]

                        point1_weighted = apply_weights(point1, weights)
                        point2_weighted = apply_weights(point2, weights)
                        point3_weighted = apply_weights(point3, weights)

                        # Calculer les vecteurs de différence
                        diff12 = point1_weighted - point2_weighted
                        diff13 = point1_weighted - point3_weighted
                        diff23 = point2_weighted - point3_weighted

                        # Calculer les distances en utilisant la norme du vecteur de différence
                        dist12 = np.linalg.norm(diff12)
                        dist13 = np.linalg.norm(diff13)
                        dist23 = np.linalg.norm(diff23)

                        if dist12 <= distance_threshold and dist13 <= distance_threshold and dist23 <= distance_threshold:
                            unified_label = f"Unified {unified_label_counter}"
                            unified_label_counter += 1
                            for label, point, updated_centers in zip((label1, label2, label3), (point1, point2, point3), (updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3)):
                                updated_centers[label] = {'point': point, 'unified_label': unified_label}


        # for label1, point1 in centers1.items():
        #     for label2, point2 in centers2.items():
        #         for label3, point3 in centers3.items():
        #             if label1 not in vectors1:
        #                 print(label1)
        #                 continue
        #             elif label2 not in vectors2:
        #                 print(label2) 
        #                 continue
        #             elif label3 not in vectors3:
        #                 print(label3)
        #                 continue

        #             # Appliquer les poids et calculer les distances
        #             point1_weighted = apply_weights(point1, weights)
        #             point2_weighted = apply_weights(point2, weights)
        #             point3_weighted = apply_weights(point3, weights)

        #             # Calculer les vecteurs de différence
        #             diff12 = point1_weighted - point2_weighted
        #             diff13 = point1_weighted - point3_weighted
        #             diff23 = point2_weighted - point3_weighted

        #             # Calculer les distances en utilisant la norme du vecteur de différence
        #             dist12 = np.linalg.norm(diff12)
        #             dist13 = np.linalg.norm(diff13)
        #             dist23 = np.linalg.norm(diff23)

        #             # Vérifier si les distances sont dans le seuil
        #             if dist12 <= distance_threshold and dist13 <= distance_threshold and dist23 <= distance_threshold:
        #                 vec1, vec2, vec3 = vectors1[label1][0], vectors2[label2][0], vectors3[label3][0]
        #                 angle12 = np.degrees(angle_between_vectors(vec1, vec2))
        #                 angle13  = np.degrees(angle_between_vectors(vec1, vec3))
        #                 angle23  = np.degrees(angle_between_vectors(vec2, vec3))

        #                 if ((angle12 < angle_threshold or abs(angle12 - 180) < angle_threshold) and (angle13 < angle_threshold or abs(angle13 - 180) < angle_threshold) and (angle23 < angle_threshold or abs(angle23 - 180) < angle_threshold)):
        #                     unified_label = f"Unified {unified_label_counter}"
        #                     unified_label_counter += 1
        #                     for label, point, updated_centers in zip((label1, label2, label3), (point1, point2, point3), (updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3)):
        #                         updated_centers[label] = {'point': point, 'unified_label': unified_label}

    validate_matches(centers_TAG1, centers_TAG2, centers_TAG3, vectors_TAG1, vectors_TAG2, vectors_TAG3, distance_threshold, angle_threshold_prim, angle_threshold_sec, weights)

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3




def keep_matching_regions_across_tags_PCA_dist(vectors_TAG1, vectors_TAG2, vectors_TAG3, df_dist1, df_dist2, df_dist3, distance_threshold=0.05, angle_threshold_prim=10, angle_threshold_sec=12):
    updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3 = {}, {}, {}
    unified_label_counter = 1

    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return np.degrees(angle)

    def validate_matches(df_dist1, df_dist2, df_dist3, vectors1, vectors2, vectors3, distance_threshold, angle_threshold_prim, angle_threshold_sec):
        nonlocal unified_label_counter
        for label1, vec1 in vectors1.items():
            rows1 = df_dist1[df_dist1['Label'] == label1]

            for label2, vec2 in vectors2.items():
                rows2 = df_dist2[df_dist2['Label'] == label2]

                for label3, vec3 in vectors3.items():
                    rows3 = df_dist3[df_dist3['Label'] == label3]

                    angles_are_similar = True
                    for (v1, v2) in [(vec1, vec2), (vec1, vec3), (vec2, vec3)]:
                        angle_prim = angle_between_vectors(v1[0], v2[0])
                        angle_sec = angle_between_vectors(v1[1], v2[1])
                        if not (angle_prim < angle_threshold_prim or abs(angle_prim - 180) < angle_threshold_prim) or not (angle_sec < angle_threshold_sec or abs(angle_sec - 180) < angle_threshold_sec):
                            angles_are_similar = False
                            break

                    if angles_are_similar:
                        distances_similar = True
                        for i in range(1, 5):  # Assuming there are 4 centre associations
                            distance1 = rows1[rows1['Centre Association'] == f'Centre {i}']['Distance'].values[0]
                            distance2 = rows2[rows2['Centre Association'] == f'Centre {i}']['Distance'].values[0]
                            distance3 = rows3[rows3['Centre Association'] == f'Centre {i}']['Distance'].values[0]
                            if not (abs(distance1 - distance2) <= distance_threshold and abs(distance1 - distance3) <= distance_threshold and abs(distance2- distance3) <= distance_threshold):
                                distances_similar = False
                                break

                        if distances_similar:
                            unified_label = f"Unified {unified_label_counter}"
                            unified_label_counter += 1
                            for label, updated_centers in zip((label1, label2, label3), (updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3)):
                                updated_centers[label] = {'unified_label': unified_label}
                            break

    validate_matches(df_dist1, df_dist2, df_dist3, vectors_TAG1, vectors_TAG2, vectors_TAG3, distance_threshold, angle_threshold_prim, angle_threshold_sec)

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3



def keep_matching_regions_across_tags_PCA_dist_same_img(pop_dico, centers_TAG1, centers_TAG2, centers_TAG3, vectors_TAG1, vectors_TAG2, vectors_TAG3, distance_threshold=0.025, angle_threshold_prim=8, angle_threshold_sec=10, weights=(0, 1, 1)):
    updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3 = {}, {}, {}
    unified_label_counter = 1

    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        angle = np.arccos(np.clip(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        return np.degrees(angle)

    def apply_weights(coordinates, weights):
        return np.multiply(coordinates, weights)

    def validate_matches(centers1, centers2, centers3, vectors1, vectors2, vectors3, distance_threshold, angle_threshold_prim, angle_threshold_sec, weights):
        nonlocal unified_label_counter
        for label1, vec1 in vectors1.items():
            match_found = False
            point1 = centers1.get(label1)
            if point1 is None: 
                continue  
            for label2, vec2 in vectors2.items():
                if match_found:
                    break
                point2 = centers2.get(label2)
                if point2 is None: 
                    continue  
                for label3, vec3 in vectors3.items():
                    point3 = centers3.get(label3)
                    if point3 is None:  # Vérifie si le point correspondant au label3 existe
                        continue  # S'il n'existe pas, passe au label3 suivant

                    angles_are_similar = True
                    # for (v1, v2) in [(vec1, vec2), (vec1, vec3), (vec2, vec3)]:
                    #     angle_prim = angle_between_vectors(v1[0], v2[0])
                    #     angle_sec = angle_between_vectors(v1[1], v2[1])
                    #     if not (angle_prim < angle_threshold_prim or abs(angle_prim - 180) < angle_threshold_prim) or not (angle_sec < angle_threshold_sec or abs(angle_sec - 180) < angle_threshold_sec):
                    #         angles_are_similar = False
                    #         break

                    if angles_are_similar:
                        point1 = centers1[label1]
                        point2 = centers2[label2]
                        point3 = centers3[label3]

                        point1_weighted = apply_weights(point1, weights)
                        point2_weighted = apply_weights(point2, weights)
                        point3_weighted = apply_weights(point3, weights)

                        diff12 = np.linalg.norm(point1_weighted - point2_weighted)
                        diff13 = np.linalg.norm(point1_weighted - point3_weighted)
                        diff23 = np.linalg.norm(point2_weighted - point3_weighted)

                        if diff12 <= distance_threshold and diff13 <= distance_threshold and diff23 <= distance_threshold:
                            unified_label = f"Unified {unified_label_counter}"
                            unified_label_counter += 1
                            for label, updated_centers in zip((label1, label2, label3), (updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3)):
                                updated_centers[label] = {'unified_label': unified_label}

                            centers1.pop(label1)
                            centers2.pop(label2)
                            centers3.pop(label3)

                            match_found = True
                            break
            if match_found:
                continue

    validate_matches(centers_TAG1, centers_TAG2, centers_TAG3, vectors_TAG1, vectors_TAG2, vectors_TAG3, distance_threshold, angle_threshold_prim, angle_threshold_sec, weights)

    for center_dico in [updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3]:
        pop_dico.append(center_dico)





def update_labels_image(labels_image, labels_to_keep):
    updated_labels_image = np.zeros_like(labels_image)

    for old_label, info in labels_to_keep.items():
        unified_label = int(info['unified_label'].split(" ")[-1])  # Extrait le numéro du label unifié
        #print(old_label)
        #old_label = int(old_label.split(" ")[-1])
        updated_labels_image[labels_image == old_label] = unified_label

    return updated_labels_image

def project_3d_to_2d_min_layers(image_3d, min_layers=2):
    """
    Projette une image 3D en 2D en ne conservant que les pixels présents dans un minimum de couches.

    Args:
    - image_3d (numpy.ndarray): Image 3D d'entrée.
    - min_layers (int): Le nombre minimum de couches où un pixel doit être présent pour être conservé.

    Returns:
    - numpy.ndarray: Image 2D projetée.
    """
    # Initialiser une image 2D avec des zéros (background)
    image_2d = np.zeros((image_3d.shape[1], image_3d.shape[2]), dtype=image_3d.dtype)

    # Compter le nombre de fois qu'un label apparaît à chaque position (x, y)
    label_count = np.zeros_like(image_2d)

    # Itérer sur chaque couche Z de l'image 3D
    for z in range(image_3d.shape[0]):
        # Itérer sur chaque pixel (x, y) de la couche
        for y in range(image_3d.shape[1]):
            for x in range(image_3d.shape[2]):
                # Si le pixel n'est pas du fond, incrémenter le compteur
                if image_3d[z, y, x] > 0:
                    label_count[y, x] += 1

                # Si le compteur atteint le nombre minimum de couches, conserver la valeur du label
                if label_count[y, x] == min_layers:
                    image_2d[y, x] = image_3d[z, y, x]

    return image_2d

def get_principal_components_2D(labels_2d_image):
    """
    Applies PCA on each label in the 2D labels image to determine the principal directions along the X and Y axes.

    Args:
    - labels_2d_image (numpy.ndarray): 2D image of labels.

    Returns:
    - dict: Dictionary containing labels as keys and the principal eigenvector as values for the principal direction in 2D.
    """
    labels = np.unique(labels_2d_image)[1:] 
    principal_components = {}

    for label in labels:
        y, x = np.where(labels_2d_image == label)
        points = np.vstack((x, y)).T 

        if points.shape[0] < 2:
            continue  

        points_centered = points - np.mean(points, axis=0)

        pca = PCA(n_components=2)  
        pca.fit(points_centered)

        principal_components[label] = [pca.components_[0], pca.components_[1]]

    return principal_components

def get_principal_components(labels_image):
    """
    Calcule les deux premiers composants principaux pour chaque label dans une image 3D.
    
    Args:
    - labels_image (numpy.ndarray): Image 3D des labels.

    Returns:
    - dict: Dictionnaire contenant les labels comme clés et les deux premiers vecteurs propres comme valeurs.
    """
    labels = np.unique(labels_image)
    principal_components = {}

    for label in labels:
        if label == 0:  # Ignorer le fond
            continue

        # Extraire les coordonnées des points correspondant au label courant
        z, y, x = np.where(labels_image == label)
        points = np.vstack((x, y, z)).T  # Empiler et transposer pour obtenir une matrice N x 3

        # Centrer les données
        points_centered = points - np.mean(points, axis=0)

        # Appliquer PCA
        pca = PCA(n_components=2)  # Utiliser 2 composants pour obtenir les deux directions principales
        pca.fit(points_centered)

        # Stocker les deux premiers vecteurs propres
        principal_components[label] = pca.components_

    return principal_components

def visualize_principal_components(image_2d, principal_components, tag_name):
    fig, ax = plt.subplots()

    ax.imshow(image_2d, cmap='gray')

    for label, vectors in principal_components.items():
        y, x = np.where(image_2d == label)
        center_x = np.mean(x)
        center_y = np.mean(y)
        ax.scatter(center_x, center_y, color='red')
        vector_scaled1 = vectors[0] * 20  # Ajustez ce facteur de mise à l'échelle au besoin
        ax.arrow(center_x, center_y, vector_scaled1[0], vector_scaled1[1], head_width=5, head_length=6, fc='red', ec='red')
        vector_scaled2 = vectors[1] * 20  # Ajustez ce facteur de mise à l'échelle au besoin
        ax.arrow(center_x, center_y, vector_scaled2[0], vector_scaled2[1], head_width=5, head_length=6, fc='blue', ec='blue')

    plt.savefig(f'/Users/xavierdekeme/Desktop/Data/PCA/PCA_{tag_name}.png') 
    plt.show()
    
def find_circles_hough(image):
    image = np.uint8(image)
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=2, minDist=75, param1=10, param2=90, minRadius=25, maxRadius=150)
    return circles

def associate_points(centers1, centers2):
    # Supposer que centers1 et centers2 sont des arrays NumPy avec une dimension inutile à l'indice 0
    centers1 = centers1[0]  # Enlever la dimension inutile
    centers2 = centers2[0]  # Enlever la dimension inutile

    # Initialisation d'un dictionnaire pour stocker les associations
    associations = {}

    for i, center1 in enumerate(centers1):
        closest_distance = np.inf
        closest_center2 = None

        for center2 in centers2:
            # Calcul de la distance euclidienne entre center1 et center2, en ne considérant que les coordonnées x et y
            distance = np.linalg.norm(center1[:2] - center2[:2])

            # Vérification si cette distance est la plus petite rencontrée jusqu'à présent
            if distance < closest_distance:
                closest_distance = distance
                closest_center2 = center2[:2]  # Mise à jour du centre le plus proche et de la distance la plus proche

        # Association du center1 avec le center2 le plus proche
        associations[f'Centre {i+1}'] = [center1[:2].tolist(), closest_center2.tolist()]

    return associations


def dict_to_o3d_point_cloud(point_dict):
    # Convertir le dictionnaire en tableau NumPy
    points = np.array(list(point_dict.values()))
    # Créer un nuage de points Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def dict_to_dataframe(point_dict):
    # Convertir le dictionnaire en DataFrame Pandas
    df = pd.DataFrame.from_dict(point_dict, orient='index', columns=['x', 'y', 'z'])
    return df

def apply_icp(source_points, target_points):
    # Convertir les dictionnaires en nuages de points Open3D
    df_source = dict_to_dataframe(source_points)
    df_target = dict_to_dataframe(target_points)

    df_source = df_source.reset_index(drop=True)
    df_target = df_target.reset_index(drop=True)

    
    # Créer des objets PointCloud pour simpleicp
    pc_fix = PointCloud(df_source, columns=["x", "y", "z"])
    pc_mov = PointCloud(df_target, columns=["x", "y", "z"])
    

    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)
    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run()

    return H, X_mov_transformed, rigid_body_transformation_params, distance_residuals


def get_normalized_center_of_labels(label_image):
    # Trouver tous les labels uniques dans l'image (en ignorant le fond qui est généralement étiqueté par 0)
    unique_labels = np.unique(label_image)[1:]  # [1:] pour exclure 0 si le fond est inclus

    # Dimensions de l'image pour la normalisation
    dims = np.array(label_image.shape)

    # Dictionnaire pour stocker les coordonnées normalisées du centre de chaque label
    normalized_centers = {}

    # Itérer sur chaque label pour calculer et normaliser son centre de masse
    for label in unique_labels:
        # Calculer le centre de masse pour le label courant
        center = np.array(center_of_mass(label_image == label))

        # Normaliser les coordonnées du centre
        normalized_center = center / dims

        # Ajouter les coordonnées normalisées au dictionnaire
        normalized_centers[label] = normalized_center

    return normalized_centers

def match_labels_general_to_tags(dict_list, weight_z=0.1):
    general_dict = dict_list[0]
    matched_labels = {}

    for general_label, general_center in general_dict.items():
        matched_tags = []  # Liste pour stocker les indices des tags correspondants

        for tag_index, tag_dict in enumerate(dict_list[1:], start=1):
            for tag_label, tag_center in tag_dict.items():
                # Appliquer un poids sur la composante z du centre
                weighted_general_center = np.array([general_center[0] * weight_z, general_center[1], general_center[2]])
                weighted_tag_center = np.array([tag_center[0] * weight_z, tag_center[1], tag_center[2]])
                
                # Vérifier la correspondance en tenant compte du poids
                if np.allclose(weighted_general_center, weighted_tag_center, atol=0.005):
                    matched_tags.append(tag_index)  # Ajouter l'indice du tag correspondant
                    break  # Passer au prochain tag_dict une fois une correspondance trouvée

        if matched_tags:
            matched_labels[general_label] = matched_tags

    return matched_labels

def eliminate_labels_from_image(image_label, center_image, pop, tag_center):
    updated_image = np.copy(image_label)

    dico_pop = pop[0]
    labels_to_eliminate = []

    for label, unified in dico_pop.items():
        distance = np.inf  # Utiliser np.inf comme valeur initiale pour la distance
        label_to_remove = None  # Initialiser à None pour vérifier plus tard si une correspondance a été trouvée
        center = np.array([tag_center[float(label)]])  # S'assurer que center est un tableau 2D

        for label_center, center_general in center_image.items():
            center_general_2d = np.array([center_general])  # Convertir en tableau 2D pour cdist
            new_distance = cdist(center, center_general_2d)

            if new_distance < distance:
                distance = new_distance
                label_to_remove = label_center

        if label_to_remove is not None:  # Vérifier si une correspondance a été trouvée
            labels_to_eliminate.append(label_to_remove)
    
    for label in labels_to_eliminate:
        updated_image[updated_image == label] = 0

    return updated_image


def apply_2d_transformation_to_3d_image(image3d, H):
    transformed_image = np.zeros_like(image3d)
    
    # Extraire la rotation et la translation de la matrice H
    # Assumons que H est de forme 3x3 avec la dernière ligne [0, 0, 1]
    rotation = H[:2, :2]
    translation = H[:2, 2]
    
    # Appliquer la transformation à chaque tranche de l'image 3D
    for z in range(image3d.shape[0]):
        transformed_image[z, :, :] = affine_transform(image3d[z, :, :], rotation, offset=translation, order=1, mode='constant', cval=0)
    
    return transformed_image

def translate_image(image, y_offset, x_offset):
    translated_image = np.zeros_like(image)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_y = y + y_offset
            new_x = x + x_offset
            
            # Vérifie si les nouvelles coordonnées sont dans les limites de l'image
            if 0 <= new_y < image.shape[0] and 0 <= new_x < image.shape[1]:
                translated_image[new_y, new_x] = image[y, x]
                    
    return translated_image

def merge_images_based_on_centers(img1, img2, center1, center2):
    z_dim = max(img1.shape[0], img2.shape[0])
    y_dim, x_dim = img1.shape[1], img1.shape[2]
    
    x_middle, y_middle = center2[0], center2[1] #utiliser le centre d'une image comme repere
    #x_middle, y_middle = x_dim //2 , y_dim//2 #utilise se le centre de l'image fictive
    x_offset_img1, x_offset_img2 = int(x_middle - center1[0] ), int(x_middle - center2[0])
    y_offset_img1, y_offset_img2 = int(y_middle - center1[1]), int(y_middle - center2[1])
    
    merged_img = np.zeros((z_dim, y_dim, x_dim), dtype=img1.dtype)
    
    mid_z_img1 = img1.shape[0] // 2
    mid_z_img2 = img2.shape[0] // 2
    mid_z_merged = z_dim // 2

    count_slice = 1
    for z in range(z_dim):
        current_slice_img1 = mid_z_img1 + z
        current_slice_img2 = mid_z_img2 + z

        if current_slice_img1 >= img1.shape[0] and current_slice_img2 >= img2.shape[0]: 
            if mid_z_img1 - count_slice >= 0:
                #slice_img1 = np.roll(np.roll(img1[mid_z_img1 - count_slice, :, :], y_offset_img1, axis=0), x_offset_img1, axis=1)
                slice_img1 = translate_image(img1[mid_z_img1 - count_slice, :, :], y_offset_img1, x_offset_img1)
            else:
                slice_img1 = np.zeros((y_dim, x_dim), dtype=img1.dtype)

            if mid_z_img2 - count_slice >= 0:
                #slice_img2 = np.roll(np.roll(img2[mid_z_img2 - count_slice, :, :], y_offset_img2, axis=0), x_offset_img2, axis=1)
                slice_img2 = translate_image(img2[mid_z_img2 - count_slice, :, :], y_offset_img2, x_offset_img2)
            else:
                slice_img2 = np.zeros((y_dim, x_dim), dtype=img2.dtype)

            merged_img[mid_z_merged - count_slice, :, :] = np.maximum.reduce([slice_img1, slice_img2])
            count_slice += 1

        else:
            if current_slice_img1 < img1.shape[0]:
                #slice_img1 = np.roll(np.roll(img1[current_slice_img1, :, :], y_offset_img1, axis=0), x_offset_img1, axis=1)
                slice_img1 = translate_image(img1[current_slice_img1, :, :], y_offset_img1, x_offset_img1)
            else:
                slice_img1 = np.zeros((y_dim, x_dim), dtype=img1.dtype)

            if current_slice_img2 < img2.shape[0]:
                #slice_img2 = np.roll(np.roll(img2[current_slice_img2, :, :], y_offset_img2, axis=0), x_offset_img2, axis=1)
                slice_img2 = translate_image(img2[current_slice_img2, :, :], y_offset_img2, x_offset_img2)
            else:
                slice_img2 = np.zeros((y_dim, x_dim), dtype=img2.dtype)

            merged_img[mid_z_merged + z, :, :] = np.maximum.reduce([slice_img1, slice_img2])
    
    return merged_img

def apply_translation_img(img, center1, center2):
    z_dim, y_dim, x_dim = img.shape[0], img.shape[1], img.shape[2]
    
    x_middle, y_middle = center2[0], center2[1]
    x_offset = int(x_middle - center1[0] )
    y_offset = int(y_middle - center1[1])
    
    merged_img = np.zeros((z_dim, y_dim, x_dim), dtype=img.dtype)

    for z in range(z_dim):
        merged_img[z, :, :] = translate_image(img[z, :, :], y_offset, x_offset)

    return merged_img

def transform_label_image(label_image, new_label_value):
    """
    Transforme les valeurs de tous les labels dans une image de label en une nouvelle valeur unique.

    Parameters:
    label_image (ndarray): Image de label d'origine.
    new_label_value (int): Nouvelle valeur à assigner à tous les labels non-zéro.

    Returns:
    ndarray: Image de label transformée.
    """
    # Créer une copie de l'image pour éviter de modifier l'originale
    transformed_image = np.copy(label_image)
    # Remplacer toutes les valeurs non-zéro par new_label_value
    transformed_image[transformed_image > 0] = new_label_value
    return transformed_image




#----------------------------------------GENERAL VARIABLES--------------------------------------------------
project_name = "Final"
general_path = "/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project" 


#----------------------------------------1st step: Get the center of each bille in each picture--------------------------------------------------
#Image1
print("Starting getting the center of each bille...")
img1_full_label_path = "/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_full_TAG123_full.tif" 
img1_2D_full = project_3d_to_2d_min_layers(imageio.volread(img1_full_label_path))
img1_circles = find_circles_hough(img1_2D_full)
print("Cirles 1: ", img1_circles)

#Image2 
img2_full_label_path = "/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_full_TAG4567_full.tif"
img2_2D_full = project_3d_to_2d_min_layers(imageio.volread(img2_full_label_path))
img2_circles = find_circles_hough(img2_2D_full)
print("Cirles 2: ", img2_circles)
print("Done getting the center of each bille!")


#----------------------------------------2nd step: Fusion of the 2 images by using the center of each bille--------------------------------------------------
print("Starting fusion of the 2 images...")
center1 = img1_circles[0][0][:2]
center2 = img2_circles[0][0][:2]

img1 = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG123_cropped_phalo.tif')
img2 = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG4567_cropped.tif')

merged_image = merge_images_based_on_centers(img1, img2, center1, center2)
imageio.volwrite('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Final/Merged.tif', merged_image)
print("Done fusion of the 2 images!")


#----------------------------------------3rd step: Get all the label images and apply the transform--------------------------------------------------
print("Starting translate the label images + getting the label centers + PCA...")
#Image1
tag1_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG1_cropped.tif')
tag1_labels = apply_translation_img(tag1_labels, center1, center2) 
tag1_centers_norm = get_normalized_center_of_labels(tag1_labels)
tag1_img_2D = project_3d_to_2d_min_layers(tag1_labels)
PCA1 = get_principal_components_2D(tag1_img_2D)
#imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/projection2D_PCA1.tif', tag1_img_2D)
#visualize_principal_components(tag1_img_2D, PCA1, "TAG1")

tag2_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG2_cropped.tif')
tag2_labels = apply_translation_img(tag2_labels, center1, center2)
tag2_centers_norm = get_normalized_center_of_labels(tag2_labels)
tag2_img_2D = project_3d_to_2d_min_layers(tag2_labels)
PCA2 = get_principal_components_2D(tag2_img_2D)
#imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/projection2D_PCA2.tif', tag2_img_2D)
#visualize_principal_components(tag2_img_2D, PCA2, "TAG2")

tag3_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG3_cropped.tif')
tag3_labels = apply_translation_img(tag3_labels, center1, center2)
tag3_centers_norm = get_normalized_center_of_labels(tag3_labels)
tag3_img_2D = project_3d_to_2d_min_layers(tag3_labels)
PCA3 = get_principal_components_2D(tag3_img_2D)
#imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/projection2D_PCA3.tif', tag3_img_2D)
#visualize_principal_components(tag3_img_2D, PCA3, "TAG3")

#Image2
tag4_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG4_cropped.tif')
tag4_centers_norm = get_normalized_center_of_labels(tag4_labels)
tag4_img_2D = project_3d_to_2d_min_layers(tag4_labels)
PCA4 = get_principal_components_2D(tag4_img_2D)
#imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/projection2D_PCA4.tif', tag4_img_2D)
#visualize_principal_components(tag4_img_2D, PCA4, "TAG4")

tag5_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG5_cropped.tif')
tag5_centers_norm = get_normalized_center_of_labels(tag5_labels)
tag5_img_2D = project_3d_to_2d_min_layers(tag5_labels)
PCA5 = get_principal_components_2D(tag5_img_2D)
# imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/projection2D_PCA5.tif', tag5_img_2D)
# visualize_principal_components(tag5_img_2D, PCA5, "TAG5")

tag6_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG6_cropped.tif')
tag6_centers_norm = get_normalized_center_of_labels(tag6_labels)
tag6_img_2D = project_3d_to_2d_min_layers(tag6_labels)
PCA6 = get_principal_components_2D(tag6_img_2D)
#imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/projection2D_PCA6.tif', tag6_img_2D)
#visualize_principal_components(tag6_img_2D, PCA6, "TAG6")

tag7_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG7_cropped.tif')
tag7_centers_norm = get_normalized_center_of_labels(tag7_labels)
tag7_img_2D = project_3d_to_2d_min_layers(tag7_labels)
PCA7 = get_principal_components_2D(tag7_img_2D)
# imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/projection2D_PCA7.tif', tag7_img_2D)
# visualize_principal_components(tag7_img_2D, PCA7, "TAG7")
print("Done translate the label images + getting the label centers + PCA!")




#----------------------------------------4rd step: Find population cells--------------------------------------------------
pop123 = []
pop567 = []
pop456 = []

pop127 = []
pop157 = []

count_pop = 1

#POP 123
population_number = "123"
population_tag = ["1", "2", "3"]
population_label = [tag1_labels, tag2_labels, tag3_labels]

keep_matching_regions_across_tags_PCA_dist_same_img(pop123, tag1_centers_norm, tag2_centers_norm, tag3_centers_norm, PCA1, PCA2, PCA3, distance_threshold=0.01, angle_threshold_prim=5, angle_threshold_sec=10)
print("--------------------------------POPULATION 123--------------------------------")
print(pop123)

for i, update_label in enumerate(pop123):
    updated_labels_images = update_labels_image(population_label[i], update_label)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/pop{population_number}/label_TAG{population_tag[i]}_pop{population_number}.tif', updated_labels_images)
    if i == 0:
        new_label_image = transform_label_image(updated_labels_images, count_pop)
        imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Final/pop{population_number}.tif', new_label_image)
        count_pop += 1


#POP 567
population_number = "567"
population_tag = ["5", "6", "7"]
population_label = [tag5_labels, tag6_labels, tag7_labels]

keep_matching_regions_across_tags_PCA_dist_same_img(pop567, tag5_centers_norm, tag6_centers_norm, tag7_centers_norm, PCA5, PCA6, PCA7, distance_threshold=0.01, angle_threshold_prim=5, angle_threshold_sec=10)
print("--------------------------------POPULATION 567--------------------------------")
print(pop567)

for i, update_label in enumerate(pop567):
    updated_labels_images = update_labels_image(population_label[i], update_label)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/pop{population_number}/label_TAG{population_tag[i]}_pop{population_number}.tif', updated_labels_images)
    if i == 0:
        new_label_image = transform_label_image(updated_labels_images, count_pop)
        imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Final/pop{population_number}.tif', new_label_image)
        count_pop += 1


#POP 456
population_number = "456"
population_tag = ["4", "5", "6"]
population_label = [tag4_labels, tag5_labels, tag6_labels]

keep_matching_regions_across_tags_PCA_dist_same_img(pop456, tag4_centers_norm, tag5_centers_norm, tag6_centers_norm, PCA4, PCA5, PCA6, distance_threshold=0.01, angle_threshold_prim=5, angle_threshold_sec=10)
print("--------------------------------POPULATION 456--------------------------------")
print(pop456)

for i, update_label in enumerate(pop456):
    updated_labels_images = update_labels_image(population_label[i], update_label)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/pop{population_number}/label_TAG{population_tag[i]}_pop{population_number}.tif', updated_labels_images)
    if i == 0:
        new_label_image = transform_label_image(updated_labels_images, count_pop)
        imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Final/pop{population_number}.tif', new_label_image)
        count_pop += 1


#POP 127
population_number = "127"
population_tag = ["1", "2", "7"]
population_label = [tag1_labels, tag2_labels, tag7_labels]

keep_matching_regions_across_tags_PCA_dist_same_img(pop127, tag1_centers_norm, tag2_centers_norm, tag7_centers_norm, PCA1, PCA2, PCA7, distance_threshold=0.01, angle_threshold_prim=5, angle_threshold_sec=10)
print("--------------------------------POPULATION 127--------------------------------")
print(pop127)

for i, update_label in enumerate(pop127):
    updated_labels_images = update_labels_image(population_label[i], update_label)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/pop{population_number}/label_TAG{population_tag[i]}_pop{population_number}.tif', updated_labels_images)
    if i == 1:
        new_label_image = transform_label_image(updated_labels_images, count_pop)
        imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Final/pop{population_number}.tif', new_label_image)
        count_pop += 1


#POP 157
population_number = "157"
population_tag = ["1", "5", "7"]
population_label = [tag1_labels, tag5_labels, tag7_labels]

keep_matching_regions_across_tags_PCA_dist_same_img(pop157, tag1_centers_norm, tag5_centers_norm, tag7_centers_norm, PCA1, PCA5, PCA7, distance_threshold=0.01, angle_threshold_prim=5, angle_threshold_sec=10)
print("--------------------------------POPULATION 157--------------------------------")
print(pop157)

for i, update_label in enumerate(pop157):
    updated_labels_images = update_labels_image(population_label[i], update_label)
    imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/pop{population_number}/label_TAG{population_tag[i]}_pop{population_number}.tif', updated_labels_images)
    if i == 1:
        new_label_image = transform_label_image(updated_labels_images, count_pop)
        imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Final/pop{population_number}.tif', new_label_image)
        count_pop += 1







#------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------BROUILLON--------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------
# index = [1, 2, 3, 4, 5, 6, 7]

# data = {}
# for i in index:
#     data[i] = {}
#     csv_centers_file_path = f'/Users/xavierdekeme/Desktop/Data/CentreNorm/centers{i}.csv'
#     csv_labels_file_path = f'/Users/xavierdekeme/Desktop/Data/Label/label{i}.csv'
#     volume_file_path = f"/Users/xavierdekeme/Desktop/Data/Volume/volume{i}.csv"

#     if i == 1 or i==2 or i==3:
#         image_shape = (14, 1024, 1024) 
#     else:
#         image_shape = (26, 1024, 1024)
#     data[i]["centers"] = read_centers_from_csv(csv_centers_file_path)
#     data[i]["labels"] = read_labels_from_csv(csv_labels_file_path, image_shape)
#     data[i]["volumes"] = read_volumes_from_csv(volume_file_path)


# tag1_centers_norm = data[1]["centers"]
# tag2_centers_norm = data[2]["centers"]
# tag3_centers_norm = data[3]["centers"]
# tag4_centers_norm = data[4]["centers"]
# tag5_centers_norm = data[5]["centers"]
# tag6_centers_norm = data[6]["centers"]
# tag7_centers_norm = data[7]["centers"]


# tag1_vol = data[1]["volumes"]
# tag2_vol = data[2]["volumes"]
# tag3_vol = data[3]["volumes"]
# tag4_vol = data[4]["volumes"]
# tag5_vol = data[5]["volumes"]
# tag6_vol = data[6]["volumes"]
# tag7_vol = data[7]["volumes"]

# tag1_labels = data[1]["labels"]
# tag2_labels = data[2]["labels"]
# tag3_labels = data[3]["labels"]
# tag4_labels = data[4]["labels"]
# tag5_labels = data[5]["labels"]
# tag6_labels = data[6]["labels"]
# tag7_labels = data[7]["labels"]



# #imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/projection2D_PCA1_full.tif', image_2D_full_TAG1)
# #PCAfull1 = get_principal_components_2D(image_2D_full_TAG1)
# #visualize_principal_components(image_2D_full_TAG1, PCAfull1, "TAG1_FULL")

# image_2D_full_TAG5 = project_3d_to_2d_min_layers(imageio.volread("/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_full_TAG5.tif"))
# circles_TAG5 = find_circles_hough(image_2D_full_TAG5)
# #imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/projection2D_PCA5_full.tif', image_2D_full_TAG5)
# #PCAfull5 = get_principal_components_2D(image_2D_full_TAG5)
# #visualize_principal_components(image_2D_full_TAG5, PCAfull5, "TAG5_FULL")


# asso = associate_points(circles_TAG1, circles_TAG5)


# df_dist1 = calculate_distances(tag1_centers_norm, asso, 0, "TAG1")
# df_dist2 = calculate_distances(tag2_centers_norm, asso, 0, "TAG2")
# df_dist3 = calculate_distances(tag3_centers_norm, asso, 0, "TAG3")
# df_dist4 = calculate_distances(tag4_centers_norm, asso, 1, "TAG4")
# df_dist5 = calculate_distances(tag5_centers_norm, asso, 1, "TAG5")
# df_dist6 = calculate_distances(tag6_centers_norm, asso, 1, "TAG6")
# df_dist7 = calculate_distances(tag7_centers_norm, asso, 1, "TAG7")






#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ICP---------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
# tag1_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG1_cropped.tif')
# tag1_centers_norm_updt = get_normalized_center_of_labels(tag1_labels)
# tag2_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG2_cropped.tif')
# tag2_centers_norm_updt = get_normalized_center_of_labels(tag2_labels)
# tag3_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG3_cropped.tif')
# tag3_centers_norm_updt = get_normalized_center_of_labels(tag3_labels)

# tag4_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG4_cropped.tif')
# tag4_centers_norm_updt = get_normalized_center_of_labels(tag4_labels)
# tag5_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG5_cropped.tif')
# tag5_centers_norm_updt = get_normalized_center_of_labels(tag5_labels)
# tag6_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG6_cropped.tif')
# tag6_centers_norm_updt = get_normalized_center_of_labels(tag6_labels)
# tag7_labels = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_TAG7_cropped.tif')
# tag7_centers_norm_updt = get_normalized_center_of_labels(tag7_labels)

# tag_cropped_img1 = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_full_TAG123_cropped.tif')
# tag_cropped_img2 = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_full_TAG4567_cropped.tif')

# image1_label = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_fused_TAG123.tif')
# center_image1 = get_normalized_center_of_labels(image1_label)

# image2_label = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_adj_fused_TAG4567.tif')
# center_image2 = get_normalized_center_of_labels(image2_label)

# dico_list = [center_image1, tag1_centers_norm, tag2_centers_norm, tag3_centers_norm]
#dico_match = match_labels_general_to_tags(dico_list)
#print(dico_match)

# dico_list2 = [center_image2, tag4_centers_norm, tag5_centers_norm, tag6_centers_norm, tag7_centers_norm]
#dico_match2 = match_labels_general_to_tags(dico_list2)
#print(dico_match2)

# updated_image1_label = eliminate_labels_from_image(tag_cropped_img1, center_image1, pop123, tag1_centers_norm_updt)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_popless_TAG123.tif', updated_image1_label)
# center_image1_popless = get_normalized_center_of_labels(updated_image1_label)

# updated_image2_label = eliminate_labels_from_image(tag_cropped_img2, center_image2, pop456, tag4_centers_norm_updt)
# updated_image2_label2 = eliminate_labels_from_image(updated_image2_label, center_image2, pop567, tag5_centers_norm_updt)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/label_popless_TAG4567.tif', updated_image2_label2)
# center_image2_popless = get_normalized_center_of_labels(updated_image2_label2)


# H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = apply_icp(center_image1_popless, center_image2_popless)
# print("H :", H)
# print("X mov :", X_mov_transformed)
# print("RGBTP :", rigid_body_transformation_params)
# print("DIST :", distance_residuals)


# image1 = imageio.imread(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG123_cropped.tif')
# image1_new = apply_2d_transformation_to_3d_image(image1, H)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Fusion/fused_TAG123_cropped_new.tif', image1_new)




# point_img1 = {}
# img1 = [tag1_centers_norm, tag2_centers_norm, tag3_centers_norm]
# point_img2 = {}
# img2 = [tag4_centers_norm, tag5_centers_norm, tag6_centers_norm, tag7_centers_norm]

# count = 1
# for i in img1:
#     for keys, values in i.items():
#         point_img1[count] = values
#         count+=1

# count = 1
# for i in img2:
#     for keys, values in i.items():
#         point_img2[count] = values
#         count+=1

# #dico_vide = {}
# #plot_3d_centers_all(point_img1, point_img2, dico_vide)


# points_A = np.array(list(point_img1.values()))
# points_B = np.array(list(point_img2.values()))


# # Utiliser la fonction ICP pour aligner points_A sur points_B
# R, t, iterations, mean_error = icp(points_A, points_B)

# print(f"Matrice de rotation estimée :\n{R}")
# print(f"Vecteur de translation estimé :\n{t}")
# print(f"Iterations : {iterations}")
# print(f"Erreur moyenne : {mean_error}")

# image = imageio.volread("/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/pop123/TAG1.tif")
# upd_img = apply_transformation(image, R, t)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/pop123/TAG1_updt.tif', upd_img)





















# update_center1, update_center2, update_center7  = keep_matching_regions_across_tags_PCA_dist(PCA1, PCA2, PCA7, df_dist1, df_dist2, df_dist7, distance_threshold=0.025)
# print("--------------------------------POPULATION 127--------------------------------")
# print(update_center1)
# print(update_center2)
# print(update_center7)

# updated_labels_images = update_labels_image(tag1_labels, update_center1)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop127/label_TAG1_update_pop127.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag2_labels, update_center2)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop127/label_TAG2_update_pop127.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag7_labels, update_center7)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop127/label_TAG7_update_pop127.tif', updated_labels_images)


# #update_center1, update_center2, update_center3  = keep_matching_regions_across_tags_PCA(tag1_centers_norm, tag2_centers_norm, tag3_centers_norm, PCA1, PCA2, PCA3)
# update_center1, update_center2, update_center3  = keep_matching_regions_across_tags_PCA_dist(PCA1, PCA2, PCA3, df_dist1, df_dist2, df_dist3, distance_threshold=0.01)
# print("--------------------------------POPULATION 123--------------------------------")
# print(update_center1)
# print(update_center2)
# print(update_center3)

# updated_labels_images = update_labels_image(tag1_labels, update_center1)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop123/label_TAG1_update_pop123.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag2_labels, update_center2)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop123/label_TAG2_update_pop123.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag3_labels, update_center3)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop123/label_TAG3_update_pop123.tif', updated_labels_images)

# #update_center1, update_center5, update_center7  = keep_matching_regions_across_tags_PCA(tag1_centers_norm, tag5_centers_norm, tag7_centers_norm, PCA1, PCA5, PCA7)
# update_center1, update_center5, update_center7  = keep_matching_regions_across_tags_PCA_dist(PCA1, PCA5, PCA7, df_dist1, df_dist5, df_dist7, distance_threshold=0.015, angle_threshold_prim=12)
# print("--------------------------------POPULATION 157--------------------------------")
# print(update_center1)
# print(update_center5)
# print(update_center7)

# updated_labels_images = update_labels_image(tag1_labels, update_center1)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop157/label_TAG1_update_pop157.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag5_labels, update_center5)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop157/label_TAG5_update_pop157.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag7_labels, update_center7)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop157/label_TAG7_update_pop157.tif', updated_labels_images)


# update_center5, update_center6, update_center7  = keep_matching_regions_across_tags_PCA_dist(PCA5, PCA6, PCA7, df_dist5, df_dist6, df_dist7, distance_threshold=0.005, angle_threshold_prim=8)
# print("--------------------------------POPULATION 567--------------------------------")
# print(update_center5)
# print(update_center6)
# print(update_center7)

# updated_labels_images = update_labels_image(tag5_labels, update_center5)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop567/label_TAG5_update_pop567.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag6_labels, update_center6)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop567/label_TAG6_update_pop567.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag7_labels, update_center7)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop567/label_TAG7_update_pop567.tif', updated_labels_images)


# update_center4, update_center5, update_center6  = keep_matching_regions_across_tags_PCA_dist(PCA4, PCA5, PCA6, df_dist4, df_dist5, df_dist6, distance_threshold=0.01)
# print("--------------------------------POPULATION 456--------------------------------")
# print(update_center4)
# print(update_center5)
# print(update_center6)

# updated_labels_images = update_labels_image(tag4_labels, update_center4)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop456/label_TAG4_update_pop456.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag5_labels, update_center5)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop456/label_TAG5_update_pop456.tif', updated_labels_images)
# updated_labels_images = update_labels_image(tag6_labels, update_center6)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Pop456/label_TAG6_update_pop456.tif', updated_labels_images)





