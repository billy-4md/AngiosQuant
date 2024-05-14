import numpy as np
import cv2
import sys
import pandas as pd
from scipy.spatial.distance import cdist
from skimage.draw import disk
import imageio
from skimage.transform import resize
import czifile
from scipy.ndimage import gaussian_filter, center_of_mass
from stardist.models import StarDist2D, StarDist3D
from csbdeep.utils import normalize
from xml.etree import ElementTree as ET
from skimage.filters import threshold_otsu
from skimage import exposure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_circle_data(csv_file):
    df = pd.read_csv(csv_file)
    # Assurez-vous de retirer les espaces avant dans les noms des colonnes
    df.columns = df.columns.str.strip()
    return df[['center x', 'center y']].values, df['radius'].values

def match_circles(circles_a, circles_b):
    distances = cdist(circles_a, circles_b)
    indices_a = np.argmin(distances, axis=1)
    return indices_a

def find_general_center(start_centers, mid_centers, end_centers, indices_first, indices_second):
    centers_global = []
    for i in range(len(mid_centers)):
        y = (mid_centers[i][0] + start_centers[indices_first[i]][0] + end_centers[indices_second[i]][0]) / 3
        x = (mid_centers[i][1] + start_centers[indices_first[i]][1] + end_centers[indices_second[i]][1]) / 3
        centers_global.append([x, y])
    return np.array(centers_global)

def interpolate_circles(start_radii, mid_radii, end_radii, n_slices):
    # Initialisez le tableau pour stocker les rayons interpolés
    interpolated_radii = np.zeros((len(start_radii), n_slices))

    # Interpolez les rayons pour chaque cercle
    for i in range(len(start_radii)):
        start = start_radii[i]
        mid = mid_radii[i]
        end = end_radii[i]

        # Interpolation linéaire pour la première moitié des tranches
        interpolated_radii[i, :n_slices//2] = np.linspace(start, mid, n_slices//2, endpoint=False)

        # Interpolation linéaire pour la deuxième moitié des tranches
        interpolated_radii[i, n_slices//2:] = np.linspace(mid, end, n_slices - n_slices//2)

    return interpolated_radii

def create_3d_image(centers_global, interpolated_radii, n_slices, image_shape):
    img_3d = np.zeros((n_slices, image_shape[0], image_shape[1]), dtype=np.uint8)
    
    for slice_idx in range(n_slices):
        for center, radius in zip(centers_global, interpolated_radii[:, slice_idx]):
            center_tuple = (int(center[0]), int(center[1]))

            # S'assurer que radius est un scalaire
            if np.isscalar(radius):
                radius_scalar = int(radius)
            elif isinstance(radius, np.ndarray) and radius.size == 1:
                radius_scalar = int(radius.item())
            else:
                raise ValueError("Radius is neither a scalar nor a single-element array.")

            rr, cc = disk(center_tuple, radius_scalar, shape=image_shape)
            img_3d[slice_idx][rr, cc] = 255

    return img_3d

#return a dictionary of the czi shape dimensions:
def dict_shape(czi):
    return dict(zip([*czi.axes],czi.shape))

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

def appliquer_filtre_gaussien_3d(image_3d, sigma=1):
    # Initialiser un tableau pour stocker l'image filtrée
    image_filtree_3d = np.zeros_like(image_3d)
    
    # Appliquer le filtre gaussien à chaque tranche Z
    for z in range(image_3d.shape[0]):
        image_filtree_3d[z, :, :] = gaussian_filter(image_3d[z, :, :], sigma=sigma)
    
    return image_filtree_3d

def channels_dict_def(metadata):
    channels = metadata.findall('.//Channel')
    count = 0 
    channels_dict = {}
    for idx, chan in enumerate(channels):
        name = chan.attrib['Name']
        if name not in channels_dict:
            channels_dict[name] = idx
    return channels_dict

def normalize_channel(channel, min_val=0, max_val=255):
    channel_normalized = exposure.rescale_intensity(channel, out_range=(min_val, max_val))
    return channel_normalized

def normalize_and_convert_to_uint8(image):
    # Assurez-vous que l'image est dans la plage de 0-255 et de type np.uint8
    image_normalized = normalize_channel(image)  # Normalise l'image avec la fonction existante
    return image_normalized.astype(np.uint8)  

def combine_normalized_channels_to_grayscale(image, dic_dim, exclude_channels=None):
    if exclude_channels is None:
        exclude_channels = []

    tot_img = np.zeros((dic_dim['Z'], dic_dim['Y'], dic_dim['X']), dtype=np.float32)

    num_channels_used = 0
    for c in range(dic_dim['C']):
        if c not in exclude_channels:
            # Extraire et normaliser le canal
            image_czi_reduite, axes_restants = czi_slicer(image, axes, indexes={'C':c})
            channel_normalized = normalize_channel(image_czi_reduite)
            tot_img += channel_normalized
            num_channels_used += 1

    # Calculer la moyenne des canaux pour obtenir l'image en échelle de gris
    if num_channels_used > 0:
        tot_img /= num_channels_used

    return tot_img.astype(np.uint16)

def isolate_and_normalize_channel(image, dic_dim, channel_index):
    if channel_index < 0 or channel_index >= dic_dim['C']:
        raise ValueError("Channel index out of range.")

    image_czi_reduite, axes_restants = czi_slicer(image, axes, indexes={'C':channel_index})

    channel_normalized = normalize_channel(image_czi_reduite)

    # Retourner le canal normalisé comme une image en niveaux de gris
    return channel_normalized.astype(np.uint16)

def get_center_of_mass(label_mask):
    return center_of_mass(label_mask)

def is_similar_center(center1, center2, threshold=5):
    return np.linalg.norm(np.array(center1) - np.array(center2)) <= threshold

def reassign_labels(labels_all_slices, df):
    new_labels_all_slices = []
    max_label = 0  # Pour garder une trace du dernier label utilisé

    for z in range(len(labels_all_slices)):
        labels = labels_all_slices[z]
        new_labels = np.zeros_like(labels)
        layer_df = df[df['Layer'] == z]

        if z == 0:
            # Première couche : réassigner les labels séquentiellement
            for idx, row in layer_df.iterrows():
                label = row['Label']
                if label == 0:
                    continue
                max_label += 1
                new_labels[labels == label] = max_label
        else:
            # Couches suivantes : comparer avec la couche précédente
            prev_layer_df = df[df['Layer'] == z - 1]
            for idx, row in layer_df.iterrows():
                label = row['Label']
                if label == 0:
                    continue

                # Trouver le label le plus proche dans la couche précédente
                current_center = np.array([[row['Center X'], row['Center Y']]])
                prev_centers = prev_layer_df[['Center X', 'Center Y']].values
                distances = cdist(current_center, prev_centers)
                min_dist_idx = np.argmin(distances)
                min_dist_label = prev_layer_df.iloc[min_dist_idx]['Label']

                seuil = 25
                if distances[0, min_dist_idx] < seuil:  # `seuil` est la distance maximale pour considérer deux centres comme similaires
                    # Assigner le même label que le label le plus proche de la couche précédente
                    new_labels[labels == label] = new_labels_all_slices[z-1][labels_all_slices[z-1] == min_dist_label][0]
                else:
                    # Sinon, créer un nouveau label
                    max_label += 1
                    new_labels[labels == label] = max_label

        new_labels_all_slices.append(new_labels)

    return np.stack(new_labels_all_slices, axis=0)

def create_mask_from_gray_image(gray_image):
    median_filtered = cv2.medianBlur(gray_image, 25)
    
    histogram = cv2.calcHist([median_filtered], [0], None, [256], [0,256])
    threshold_value, _ = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    mask = (median_filtered >= threshold_value).astype(np.uint8)
    
    return mask

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

with czifile.CziFile('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Merge/1.IF1.czi') as czi:
    image_czi = czi.asarray()
    dic_dim = dict_shape(czi)  
    axes = czi.axes  
    metadata_xml = czi.metadata()
    metadata = ET.fromstring(metadata_xml)

channels_dict = channels_dict_def(metadata)
indice_canal_a_exclure = channels_dict['ChS1-T3'] #PHALLOIDINE
indice_canal_to_keep = channels_dict['Ch2-T1'] #TAG 1 

combined_image = combine_normalized_channels_to_grayscale(image_czi, dic_dim, exclude_channels=[indice_canal_a_exclure])
isolate_image = isolate_and_normalize_channel(image_czi, dic_dim, indice_canal_to_keep)
#imageio.volwrite('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Merge/3d_image_combined_2.tif', combined_image)
imageio.volwrite('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Merge/3d_image_isolate_TAG_1.tif', isolate_image)


# Charger le modèle pré-entraîné StarDist 2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

labels_all_slices = []
data = []
centers_dict = {z: [] for z in range(dic_dim['Z'])}

for z in range(dic_dim['Z']):
    img_slice = isolate_image[z, :, :]
    img_slice = normalize(img_slice)
    labels, details = model.predict_instances(img_slice)
    unique_labels = np.unique(labels)

    SURFACE_THRESHOLD_SUP = 550 
    SURFACE_THRESHOLD_INF = 25 

    labels_to_remove = []
    for label_num in unique_labels:
        if label_num == 0: 
            continue

        instance_mask = labels == label_num
        label_surface = np.sum(instance_mask)

        if label_surface > SURFACE_THRESHOLD_SUP or label_surface < SURFACE_THRESHOLD_INF:
            labels[instance_mask] = 0
            labels_to_remove.append(label_num)
        else:
            center = center_of_mass(instance_mask)
            centers_dict[z].append(center)


    unique_labels = np.array([label for label in unique_labels if label not in labels_to_remove])

    for label in unique_labels:
        if label == 0:  
            continue
        center = center_of_mass(labels == label)
        data.append({'Layer': z, 'Label': label, 'Center X': center[0], 'Center Y': center[1]})

    labels_all_slices.append(labels)


df = pd.DataFrame(data)
df.to_csv('/Users/xavierdekeme/Desktop/segmentation_details.csv', index=False)
labels_all_slices = np.stack(labels_all_slices, axis=0)
labels_all_slices_updated = reassign_labels(labels_all_slices, df)

centers = find_centers_of_labels_in_3d(labels_all_slices_updated)
plot_3d_centers(centers)

# # Sauvegarder l'image CZI modifiée
# imageio.volwrite('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/hough_transform/3d_image_final_test10.tif', image_czi_reduite)
imageio.volwrite('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/Merge/3d_image_label_iso_TAG1.tif', labels_all_slices_updated)



