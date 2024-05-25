import os
import json
import numpy as np
import pandas as pd
import cv2
import imageio

from scipy.ndimage import median_filter, zoom, rotate
from skimage.transform import resize

import utils
import segmentation_tool


"""-------------HOUGH FUNCTIONS----------"""
def find_circles(composite_image):  
    layer_to_keep = 4    
    segmentation_thresholds = (1000000, 10)

    mid_layer = composite_image.shape[0] // 2
    focused_image = composite_image[mid_layer-layer_to_keep:mid_layer+layer_to_keep, :, :]

    labels = segmentation_tool.do_segmentation_StarDist(focused_image, *segmentation_thresholds)

    img_2D_full = project_3d_to_2d_min_layers(labels)
    circles = hough_transform(img_2D_full)

    return circles

def project_3d_to_2d_min_layers(image_3d, min_layers=2, median_size=5, closing_size=3):
    """
    Projette une image 3D en 2D en ne conservant que les pixels présents dans un minimum de couches,
    avec traitement par filtre médian et fermeture morphologique pour améliorer la qualité.

    Args:
    - image_3d (numpy.ndarray): Image 3D d'entrée.
    - min_layers (int): Le nombre minimum de couches où un pixel doit être présent pour être conservé.
    - median_size (int): Taille du voisinage pour le filtre médian.
    - closing_size (int): Taille de l'élément structurant pour la fermeture morphologique.

    Returns:
    - numpy.ndarray: Image 2D projetée après traitement.
    """
    # Count how many layers each pixel is present in
    label_count = np.count_nonzero(image_3d > 0, axis=0)
    mask = label_count >= min_layers
    image_2d = np.zeros((image_3d.shape[1], image_3d.shape[2]), dtype=image_3d.dtype)

    # Apply projection logic
    for z in range(image_3d.shape[0]):
        update_mask = (image_2d == 0) & mask & (image_3d[z] > 0)
        image_2d[update_mask] = image_3d[z][update_mask]

    # Apply a median filter to reduce noise
    image_2d = median_filter(image_2d, size=median_size)

    # Apply morphological closing to fill gaps and connect nearby objects
    #selem = disk(closing_size)  # Create a disk-shaped structural element
    #image_2d = binary_closing(image_2d, selem)

    # Save the processed image
    #imageio.imwrite(os.path.join(project_path, "img_2d.tif"), image_2d.astype(np.uint8))

    return image_2d
    
def hough_transform(image):
    image = np.uint8(image)
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=2, minDist=200, param1=10, param2=50, minRadius=250, maxRadius=450)
    return circles


"""-------------CONTROL POINTS/VECTORS FUNCTIONS----------"""
def load_control_points(file_path):
    """
    Charge les points de contrôle à partir d'un fichier .txt.
    """
    control_points = []
    with open(file_path, 'r') as file:
        for line in file:
            point = tuple(map(float, line.strip().split(',')))
            control_points.append(point)
    return np.array(control_points)

def find_closest_labels(centers, control_points, z_weight=0.5):
    """
    Trouve le numéro de label du centre de nucléus le plus proche de chaque point de contrôle.
    """
    matched_labels = []

    for point in control_points:
        z, y, x = point
        distances = []
        
        for label, (cz, cy, cx) in centers.items():
            distance = np.sqrt((z_weight * (z - cz))**2 + (y - cy)**2 + (x - cx)**2)
            distances.append((distance, label))
        
        closest_label = min(distances, key=lambda t: t[0])[1]
        matched_labels.append(closest_label)
    
    return matched_labels

def calculate_vectors(label_list, center_bille, image_name, z_dim, tag_center):
    """
    Calcule les vecteurs de chaque centre par rapport au centre de la bille.
    """
    vectors = {}
    for label in label_list:
        z, y, x = tag_center[image_name][label]
        vector = np.array([z - z_dim, y - center_bille[0], x - center_bille[1]])
        #normalized_vector = vector / np.linalg.norm(vector)
        vectors[label] = vector
    return vectors

def find_vector_matches(vectors1, vectors2, threshold=0.95):
    """
    Trouve les correspondances entre les vecteurs similaires dans les deux ensembles.
    La correspondance est basée sur le produit scalaire entre les vecteurs.
    """
    matches = {}
    for label1, vec1 in vectors1.items():
        best_match = None
        best_similarity = -1
        
        for label2, vec2 in vectors2.items():
            similarity = np.dot(vec1, vec2)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = label2
        
        if best_similarity >= threshold:
            matches[label1] = best_match
    
    return matches

def calculate_scaling_factor(matches, center_bille1, center_bille2, z_dim1, z_dim2, tag_center):
    """
    Calcule le facteur d'échelle entre deux images en utilisant les distances par rapport aux centres de la bille.
    """
    scaling_factors = []
    
    for label1, label2 in matches.items():
        z1, y1, x1 = tag_center["image1"][label1]
        z2, y2, x2 = tag_center["image2"][label2]
        
        vec1 = np.array([z1 - z_dim1, y1 - center_bille1[0], x1 - center_bille1[1]])
        vec2 = np.array([z2 - z_dim2, y2 - center_bille2[0], x2 - center_bille2[1]])
        
        distance1 = np.linalg.norm(vec1)
        distance2 = np.linalg.norm(vec2)
        
        if distance1 != 0:
            scaling_factor = distance2 / distance1
            scaling_factors.append(scaling_factor)
    
    scaling_factor = np.mean(scaling_factors)
    print(f"Scaling factor: {scaling_factor}")
    return scaling_factor

def calculate_angle(v):
    """
    Calcule l'angle en radians du vecteur donné par rapport à l'axe des x.
    """
    return np.arctan2(v[1], v[0])

def calculate_rotation_angle(matches, center_bille1, center_bille2, tag_center):
    """
    Calcule l'angle de rotation moyen entre deux ensembles de points en utilisant leurs correspondances
    et leurs positions relatives au centre de la bille.
    """
    rotation_angles = []
    
    for label1, label2 in matches.items():
        z1, y1, x1 = tag_center["image1"][label1]
        z2, y2, x2 = tag_center["image2"][label2]
        
        vec1 = np.array([y1 - center_bille1[0], x1 - center_bille1[1]])
        vec2 = np.array([y2 - center_bille2[0], x2 - center_bille2[1]])
        
        norm_vec1 = vec1 / np.linalg.norm(vec1) if np.linalg.norm(vec1) != 0 else vec1
        norm_vec2 = vec2 / np.linalg.norm(vec2) if np.linalg.norm(vec2) != 0 else vec2
        
        angle1 = calculate_angle(norm_vec1)
        angle2 = calculate_angle(norm_vec2)
        
        angle_difference = angle2 - angle1
        rotation_angles.append(angle_difference)
    
    average_angle = np.degrees(np.mean(rotation_angles))
    print(f"Rotation angle: {average_angle}")

    return average_angle
    

"""-------------TRANSFORMATION FUNCTIONS----------"""
def translate_image(image, y_offset, x_offset, scaling_factor, scaling = True):
    if scaling:
        scaled_image = zoom(image, (scaling_factor, scaling_factor), order=0)
    else:
        scaled_image = image

    translated_image = np.zeros_like(scaled_image)
    
    for y in range(scaled_image.shape[0]):
        for x in range(scaled_image.shape[1]):
            new_y = y + y_offset
            new_x = x + x_offset
            
            if 0 <= new_y < scaled_image.shape[0] and 0 <= new_x < scaled_image.shape[1]:
                translated_image[new_y, new_x] = scaled_image[y, x]
                    
    return translated_image

def merge_images(img1_raw, img2_raw, center1, center2, scale_factor, angle, axes=(1, 0)):
    img1, img2 = apply_transfo_img(img1_raw, img2_raw, center1, center2, scale_factor)

    z_dim, y_dim, x_dim = min(img1.shape[0], img2.shape[0]), img1.shape[1], img1.shape[2]
    merged_img = np.zeros((z_dim, y_dim, x_dim), dtype=img1.dtype)

    if img1.shape[0] < img2.shape[0]:
        diff_slice = img2.shape[0] - img1.shape[0]
        img2 = img2[diff_slice//2:-diff_slice//2, :, :]
    elif img1.shape[0] > img2.shape[0]:
        diff_slice = img1.shape[0] - img2.shape[0]
        img1 = img1[diff_slice//2:-diff_slice//2, :, :]


    for z in range(z_dim):
        slice_img1 = img1[z, :, :]
        slice_img2 = img2[z, :, :]

        rotated_img = rotate(slice_img1, angle, reshape=False, axes=axes)
        slice_img1 = resize(rotated_img, (y_dim, x_dim), order=0, mode='edge', anti_aliasing=False)

        merged_img[z, :, :] = np.maximum(slice_img1, slice_img2)

    utils.save_image_function(merged_img, 'merged_img.tif', project_path)

    return merged_img

def apply_transfo_img(img1, img2, center_img1, center_img2, scaling_factor, image_to_save=True):
    x1, y1 = center_img1
    x2, y2 = center_img2

    x_middle, y_middle = img1.shape[2]//2, img1.shape[1]//2 
    x_offset_img1, x_offset_img2  = int(x_middle - x1), int(x_middle - x2)
    y_offset_img1, y_offset_img2 = int(y_middle - y1), int(y_middle - y2)

    img1_mod = np.zeros((img1.shape[0], img1.shape[1], img1.shape[2]), dtype=img1.dtype)
    img2_mod = np.zeros((img2.shape[0], img2.shape[1], img2.shape[2]), dtype=img2.dtype)

    croppage_value_x = int(((img1.shape[2] * scaling_factor) - img1.shape[2]) / 2)
    croppage_value_y = int(((img1.shape[1] * scaling_factor) - img1.shape[1]) / 2)

    for z in range(img1.shape[0]):
        slice_img1 = img1[z, :, :]
        slice_img1 = translate_image(slice_img1, y_offset_img1, x_offset_img1, scaling_factor)
        slice_img1 = resize(slice_img1[croppage_value_y:-croppage_value_y, croppage_value_x:-croppage_value_x], (img1.shape[1], img1.shape[2]), order=0, mode='edge', anti_aliasing=False)
        img1_mod[z, :, :] = slice_img1

    for z in range(img2.shape[0]):
        slice_img2 = img2[z, :, :]
        slice_img2 = translate_image(slice_img2, y_offset_img2, x_offset_img2, scaling_factor, scaling=False) #Scaling = False allow to only make the translation and not the scaling
        img2_mod[z, :, :] = slice_img2

    if image_to_save:
        utils.save_image_function(img1_mod, f'image1_processed.tif', project_path)
        utils.save_image_function(img2_mod, f'image2_processed.tif', project_path)

    return img1_mod, img2_mod


"""-------------MAIN----------"""
def main_semi_auto_merge(current_project):
    global json_path, project_path
    global cellpose_max_th, cellpose_min_th, cellpose_diam_value
    global startdist_max_th, stardist_min_th

    #Global variable
    resources_path = os.environ.get('FLASK_RESOURCES_PATH', os.getcwd())
    project_path = os.path.join(resources_path, 'python', 'server', 'project', current_project) 
    json_path = os.path.join(resources_path, 'python', 'server', 'json') 
    tag_center = {}
    image_list = ["image1", "image2"]
    cellpose_max_th = 10000
    cellpose_min_th = 1000
    cellpose_diam_value = 60
    startdist_max_th = 4500
    stardist_min_th = 1500

    #Load json data
    json_file_path = os.path.join(json_path, 'settings_data.json')
    with open(json_file_path, 'r') as f:
        tags_info = json.load(f) 
    utils.modify_running_process("Semi auto merging running", json_path)

    #Creation of needed variables/images/...
    for image in image_list:
        utils.create_full_image(image, tags_info, project_path)

    #Load control points:
    control_points_path1 = os.path.join(project_path, "control_point1.txt") 
    control_points_path2 = os.path.join(project_path, "control_point2.txt")  
    control_points1 = load_control_points(control_points_path1)
    control_points2 = load_control_points(control_points_path2)
    if (len(control_points1) == 0 and len(control_points2) == 0) or (len(control_points1) != len(control_points2)): #If problems in controls points, stop the program
        utils.modify_running_process("Need to complete the control points to launch the semi-automatic", json_path)
        return 
    
    image1_composite = imageio.volread(os.path.join(project_path, "full_image1.tif"))
    image2_composite = imageio.volread(os.path.join(project_path, "full_image2.tif"))

    #1st: Find center with Hough Transform
    center_img1 = find_circles(image1_composite)[0][0][:2] #Coordinates of the center of the bead (only y and x)
    center_img2_gene = find_circles(image2_composite)
    center_img2, radius2 = center_img2_gene[0][0][:2], center_img2_gene[0][0][2]

    #2nd: Apply full segmentation on each image and get normalized center
    full_label_img1 = segmentation_tool.do_segmentation_StarDist(image1_composite, startdist_max_th, stardist_min_th)
    full_label_img2 = segmentation_tool.do_segmentation_StarDist(image2_composite, startdist_max_th, stardist_min_th)
    utils.save_image_function(full_label_img1, "label_img1.tif", project_path)
    utils.save_image_function(full_label_img2, "label_img2.tif", project_path)

    #3rd: Find matching point between control points and nuclei center for each image
    tag_center["image1"] = segmentation_tool.get_center_of_labels(full_label_img1)
    tag_center["image2"] = segmentation_tool.get_center_of_labels(full_label_img2)
    
    closest_labels1 = find_closest_labels(tag_center["image1"], control_points1) #List with the label values of the matching nucleus/control point
    closest_labels2 = find_closest_labels(tag_center["image2"], control_points2)

    #4th: Calculate vector for each nucleus image and find corresponding vector to match the nucleus between the 2 images
    z_dim1 = image1_composite.shape[0]
    z_dim2 = image2_composite.shape[0]
    vectors_image1 = calculate_vectors(closest_labels1, center_img1, "image1", z_dim1, tag_center)
    vectors_image2 = calculate_vectors(closest_labels2, center_img2, 'image2', z_dim2, tag_center)
    matches = find_vector_matches(vectors_image1, vectors_image2)
    #match_label_img1, match_label_img2 = segmentation_tool.unify_labels(full_label_img1, full_label_img2, matches)
    #utils.save_image_function(match_label_img1, "match_label_img1.tif", project_path)
    #utils.save_image_function(match_label_img2, "match_label_img2.tif", project_path)

    #5th: Calculate the scaling/rotation between the 2 images + transformation and merging
    scaling_factor = calculate_scaling_factor(matches, center_img1, center_img2, z_dim1, z_dim2, tag_center)
    rotation_angle = calculate_rotation_angle(matches, center_img1, center_img2, tag_center)
   
    merged_img = merge_images(image1_composite, image2_composite, center_img1, center_img2, scaling_factor, rotation_angle) 
   
    #Ending the program and saving all the necessary info
    data_project = {
        "scale_factor": str(scaling_factor),
        "rotation_factor": str(rotation_angle),
        "img1_processing": f"{tags_info['image1']['upperLayersToRemove']}, {tags_info['image1']['lowerLayersToRemove']}",
        "img2_processing": f"{tags_info['image2']['upperLayersToRemove']}, {tags_info['image2']['lowerLayersToRemove']}",
        "center_bead1": f"{center_img1[0]}, {center_img1[1]}",
        "center_bead2": f"{center_img2[0]}, {center_img2[1]}",
        "radius_bead2": str(radius2),
        "cropping_value": str(tags_info["croppingValue"]),
        "phalo_tag": str(tags_info["phaloTag"]),
        "last_merging_run": "Semi-Automatic merging"       
    }
    utils.save_project_info(data_project, project_path)
    utils.modify_running_process("Done Merging Semi-Automatic", json_path)





