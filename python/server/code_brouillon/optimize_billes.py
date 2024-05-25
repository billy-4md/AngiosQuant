import numpy as np
import re
import cv2
import sys
import pandas as pd
import czifile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

from scipy.spatial.distance import cdist
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

def plot_3d_centers_all(centers_TAG1, centers_TAG2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Filtrer et tracer les centres de TAG1
    filtered_TAG1 = [center for center in centers_TAG1.values() if 0.25 <= center[1] <= 0.75 and 0.25 <= center[2] <= 0.75]
    if filtered_TAG1:  # Vérifier si la liste n'est pas vide
        xs_TAG1, ys_TAG1, zs_TAG1 = zip(*filtered_TAG1)  # Déballer les listes filtrées
        ax.scatter(xs_TAG1, ys_TAG1, zs_TAG1, color='r', label='TAG1')
        np.savez('/Users/xavierdekeme/Desktop/Data/PointCloud/IMG1_centers.npz', xs=xs_TAG1, ys=ys_TAG1, zs=zs_TAG1)

    # Filtrer et tracer les centres de TAG2
    filtered_TAG2 = [center for center in centers_TAG2.values() if 0.25 <= center[1] <= 0.75 and 0.25 <= center[2] <= 0.75]
    if filtered_TAG2:  # Vérifier si la liste n'est pas vide
        xs_TAG2, ys_TAG2, zs_TAG2 = zip(*filtered_TAG2)  # Déballer les listes filtrées
        ax.scatter(xs_TAG2, ys_TAG2, zs_TAG2, color='b', label='TAG2')
        np.savez('/Users/xavierdekeme/Desktop/Data/PointCloud/IMG2_centers.npz', xs=xs_TAG2, ys=ys_TAG2, zs=zs_TAG2)

    ax.set_xlabel('Z Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('X Coordinate')
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
        points = np.vstack((x, y)).T  # Stack and transpose to get an N x 2 matrix

        if points.shape[0] < 2:
            continue  

        points_centered = points - np.mean(points, axis=0)

        pca = PCA(n_components=2)  # Use 1 component to obtain the principal direction in 2D
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
        # vector_scaled1 = vectors[0] * 20  # Ajustez ce facteur de mise à l'échelle au besoin
        # ax.arrow(center_x, center_y, vector_scaled1[0], vector_scaled1[1], head_width=5, head_length=6, fc='red', ec='red')
        # vector_scaled2 = vectors[1] * 20  # Ajustez ce facteur de mise à l'échelle au besoin
        # ax.arrow(center_x, center_y, vector_scaled2[0], vector_scaled2[1], head_width=5, head_length=6, fc='blue', ec='blue')

    plt.savefig(f'/Users/xavierdekeme/Desktop/Data/PCA/dot_{tag_name}.png') 
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


def calculate_distances(tag_centers, associations, tag_index, tag_name):
    # Initialiser une liste pour stocker les données avant de créer le DataFrame
    data = []

    def euclidean_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # Itérer sur chaque centre du TAG
    for label, tag_center in tag_centers.items():
        for key, value in associations.items():
            new_point = [tag_center[2], tag_center[1]] #x, y norm
            distance = euclidean_distance(new_point, [value[tag_index][0]/1024, value[tag_index][1]/1024])
            # Ajouter les données dans la liste au lieu de les append au DataFrame
            data.append({'Label': label, 'Centre Association': key, 'Distance': distance})

    # Créer le DataFrame à partir de la liste complète de données
    distances_df = pd.DataFrame(data)

    # Sauvegarder le DataFrame en CSV
    distances_df.to_csv(f'/Users/xavierdekeme/Desktop/Data/Distance/{tag_name}_norm.csv', index=False)

    return distances_df


index = [1, 2]

data = {}
for i in index:
    data[i] = {}
    csv_centers_file_path = f'/Users/xavierdekeme/Desktop/Data/CentreNorm/centers_billes{i}.csv'
    csv_labels_file_path = f'/Users/xavierdekeme/Desktop/Data/Label/label_billes{i}.csv'
    volume_file_path = f"/Users/xavierdekeme/Desktop/Data/Volume/volume_billes{i}.csv"

    if i == 1:
        image_shape = (154, 1024, 1024) 
    else:
        image_shape = (161, 1024, 1024)
    data[i]["centers"] = read_centers_from_csv(csv_centers_file_path)
    data[i]["labels"] = read_labels_from_csv(csv_labels_file_path, image_shape)
    data[i]["volumes"] = read_volumes_from_csv(volume_file_path)


tag1_centers_norm = data[1]["centers"]
tag2_centers_norm = data[2]["centers"]

plot_3d_centers_all(tag1_centers_norm, tag2_centers_norm)

# Charger les données à partir des fichiers .npz
data_tag1 = np.load('/Users/xavierdekeme/Desktop/Data/PointCloud/IMG1_centers.npz')
data_tag2 = np.load('/Users/xavierdekeme/Desktop/Data/PointCloud/IMG2_centers.npz')

# Extraire les coordonnées x, y, z pour chaque ensemble de points
xs_TAG1, ys_TAG1, zs_TAG1 = data_tag1['xs'], data_tag1['ys'], data_tag1['zs']
xs_TAG2, ys_TAG2, zs_TAG2 = data_tag2['xs'], data_tag2['ys'], data_tag2['zs']

# Convertir les coordonnées en tableaux NumPy pour faciliter le calcul
points_array_tag1 = np.vstack((zs_TAG1, ys_TAG1, xs_TAG1)).T  # Notez l'ordre (z, y, x) pour correspondre à votre format de données
points_array_tag2 = np.vstack((zs_TAG2, ys_TAG2, xs_TAG2)).T

# Calculer le barycentre pour le premier ensemble de points
centre_x_tag1 = np.mean(points_array_tag1[:, 2])
centre_y_tag1 = np.mean(points_array_tag1[:, 1])
centre_z_tag1 = np.mean(points_array_tag1[:, 0])

# Créer un dictionnaire pour le centre du premier ensemble de points
dico_center1 = {"centre": [centre_z_tag1*154, centre_y_tag1*1024, centre_x_tag1*1024]}
print("Centre du premier ensemble de points :", dico_center1)

# Calculer le barycentre pour le deuxième ensemble de points
centre_x_tag2 = np.mean(points_array_tag2[:, 2])
centre_y_tag2 = np.mean(points_array_tag2[:, 1])
centre_z_tag2 = np.mean(points_array_tag2[:, 0])

# Créer un dictionnaire pour le centre du deuxième ensemble de points
dico_center2 = {"centre": [centre_z_tag2*161, centre_y_tag2*1024, centre_x_tag2*1024]}
print("Centre du deuxième ensemble de points :", dico_center2)


# tag1_vol = data[1]["volumes"]
# tag2_vol = data[2]["volumes"]

# tag1_labels = data[1]["labels"]
# tag2_labels = data[2]["labels"]


#plot_3d_centers_all(tag1_centers_norm, tag2_centers_norm)
#plot_3d_centers_all(tag2_centers_norm, dico_center2, dico_vide2)

# image_2D_tag1 = project_3d_to_2d_min_layers(tag1_labels)
# image_2D_tag2 = project_3d_to_2d_min_layers(tag2_labels)


# PCA1 = get_principal_components_2D(image_2D_tag1)
# imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/billes/projection2D_DAPI_IMG1.tif', image_2D_tag1)
# visualize_principal_components(image_2D_tag1, PCA1, "DAPI_IMG1")

# PCA2 = get_principal_components_2D(image_2D_tag2)
# imageio.imwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/billes/projection2D_DAPI_IMG2.tif', image_2D_tag2)
# visualize_principal_components(image_2D_tag2, PCA2, "DAPI_IMG2")





