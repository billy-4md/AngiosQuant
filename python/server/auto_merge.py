import os
import json
import numpy as np
import pandas as pd
import imageio
import cv2

from scipy.ndimage import median_filter, zoom, rotate
from skimage.transform import resize
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import utils
import segmentation_tool 


"""-------------HOUGH FUNCTIONS----------"""
def find_circles(img):  
    layer_to_keep = 4    
    segmentation_thresholds = (1000000, 10)

    mid_layer = img.shape[0] // 2
    print(mid_layer)
    focused_image = img[mid_layer-layer_to_keep:mid_layer+layer_to_keep, :, :]

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

    return image_2d
    
def hough_transform(image):
    image = np.uint8(image)
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=2, minDist=200, param1=10, param2=50, minRadius=250, maxRadius=450)
    return circles

def calcul_scale_factor(radius1, radius2):
    scale_factor = radius2 / radius1 if radius1 > 0 else 1
    print(f"Scale factor: {scale_factor}")

    return scale_factor


"""-------------TRANSFO FUNCTIONS----------"""
def translate_image(image, y_offset, x_offset, scale_factor, value = True):
    if value:
        scaled_image = zoom(image, (scale_factor, scale_factor), order=0)
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

def apply_transfo_img(img1, img2, center1, center2, scale_factor):
    x_middle, y_middle = img1.shape[2]//2, img1.shape[1]//2 #Center of the image to translate
    x_offset_img1, x_offset_img2  = int(x_middle - center1[0]), int(x_middle - center2[0])
    y_offset_img1, y_offset_img2 = int(y_middle - center1[1]), int(y_middle - center2[1])

    img1_mod = np.zeros((img1.shape[0], img1.shape[1], img1.shape[2]), dtype=img1.dtype)
    img2_mod = np.zeros((img2.shape[0], img2.shape[1], img2.shape[2]), dtype=img2.dtype)

    croppage_value_x = int(((img1.shape[2] * scale_factor) - img1.shape[2]) / 2)
    croppage_value_y = int(((img1.shape[1] * scale_factor) - img1.shape[1]) / 2)
    for z in range(img1.shape[0]):
        slice_img1 = img1[z, :, :]
        slice_img1 = translate_image(slice_img1, y_offset_img1, x_offset_img1, scale_factor)
        slice_img1 = resize(slice_img1[croppage_value_y:-croppage_value_y, croppage_value_x:-croppage_value_x], (img1.shape[1], img1.shape[2]), order=0, mode='edge', anti_aliasing=False)
        img1_mod[z, :, :] = slice_img1
    for z in range(img2.shape[0]):
        slice_img2 = img2[z, :, :]
        slice_img2 = translate_image(slice_img2, y_offset_img2, x_offset_img2, scale_factor, value=False) #Value = False allow to only make the translation and not the scaling
        img2_mod[z, :, :] = slice_img2

    utils.save_image_function(img1_mod, f'image1_processed.tif', project_path)
    utils.save_image_function(img2_mod, f'image2_processed.tif', project_path)

    labels_img1 = segmentation_tool.do_segmentation_CellPose(img1_mod, cellpose_max_th, cellpose_min_th, cellpose_diam_value)
    labels_img2 = segmentation_tool.do_segmentation_CellPose(img2_mod, cellpose_max_th, cellpose_min_th, cellpose_diam_value)

    return labels_img1, labels_img2

def apply_transfo_on_label(label_img1, label_img2, center_img1, center_img2, scaling_factor, do_rotation=False):
    """
    Applique une mise à l'échelle et une rotation aux images étiquetées.
    Effectue une rotation autour de l'axe Z en faisant pivoter chaque tranche.
    """
    label_img1_mod, label_img2_mod = apply_transfo_img(label_img1, label_img2, center_img1, center_img2, scaling_factor, image_to_save=False)

    if do_rotation:
        for z in range(label_img1_mod.shape[0]):
            label_img1_mod[z, :, :] = rotate(label_img1_mod[z, :, :], angles[0], reshape=False, mode='constant', cval=0)

    utils.save_image_function(label_img1_mod, "label_image1_processed.tif", project_path)
    utils.save_image_function(label_img2_mod, "label_image2_processed.tif", project_path)

    return label_img1_mod, label_img2_mod

def start_merging_process(tag_center, z_weight=0.9):
    global angles

    center_img1 = tag_center["image1_processed"]
    center_img2 = tag_center["image2_processed"]
    
    pt_cloud1 = []
    pt_cloud2 = []

    for (label1, center1), (label2, center2) in zip(center_img1.items(), center_img2.items()):
        weighted_center1 = np.array([center1[0], center1[1], center1[2] * z_weight])
        weighted_center2 = np.array([center2[0], center2[1], center2[2] * z_weight])
        pt_cloud1.append(weighted_center1)
        pt_cloud2.append(weighted_center2)

    angles = start_icp(pt_cloud1, pt_cloud2)
    print(f"Valua angle rotation {angles}")
    merge_images(angles)

def prepare_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def perform_basic_icp(source_pcd, target_pcd):
    trans_init = np.eye(4)  

    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, 0.1, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def start_icp(points1, points2):
    pcd1 = prepare_point_cloud(points1)
    pcd2 = prepare_point_cloud(points2)

    result_basic = perform_basic_icp(pcd1, pcd2)
    rotation_matrix = np.array(result_basic.transformation[:3, :3], dtype=np.float64, copy=True)
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True) 

    return euler_angles

def merge_images(angle, axes=(1, 0)):
    img1 = imageio.volread(os.path.join(project_path, "image1_processed.tif"))
    img2 = imageio.volread(os.path.join(project_path, "image2_processed.tif"))

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

        rotated_img = rotate(slice_img1, angle[0], reshape=False, axes=axes)
        slice_img1 = resize(rotated_img, (y_dim, x_dim), order=0, mode='edge', anti_aliasing=False)

        merged_img[z, :, :] = np.maximum(slice_img1, slice_img2)
  
    utils.save_image_function(merged_img, 'merged_img.tif', project_path)



"""-------------MAIN----------"""
def main_auto_merge(current_project):
    global project_path, resources_path, json_path
    global cellpose_max_th, cellpose_min_th, cellpose_diam_value

    #Global variable
    resources_path = os.environ.get('FLASK_RESOURCES_PATH', os.getcwd())
    project_path = os.path.join(resources_path, 'python', 'server', 'project', current_project) 
    json_path = os.path.join(resources_path, 'python', 'server', 'json') 
    image_list = ["image1", "image2"]
    cellpose_max_th = 10000
    cellpose_min_th = 1000
    cellpose_diam_value = 60

    #Load json data
    json_file_path = os.path.join(json_path, 'settings_data.json')
    with open(json_file_path, 'r') as f:
        tags_info = json.load(f)

    try:
        #Modify the json file (To put the process running inside the application)   
        utils.modify_running_process("Merging auto running", json_path)

        #Creation of the full image
        for image in image_list:
            utils.create_full_image(image, tags_info, project_path)

        full_img1 = imageio.volread(os.path.join(project_path, "full_image1.tif"))
        full_img2 = imageio.volread(os.path.join(project_path, "full_image2.tif"))

        circles1 = find_circles(full_img1)
        circles2 = find_circles(full_img2)
        radius1, radius2 = circles1[0][0][2], circles2[0][0][2]
        center1, center2 = circles1[0][0][:2], circles2[0][0][:2]
        scale_factor = calcul_scale_factor(radius1, radius2)

        tag_center = {}
        full_processed_label_img1, full_processed_label_img2 = apply_transfo_img(full_img1, full_img2 ,center1, center2, scale_factor)
        tag_center["image1_processed"] = segmentation_tool.get_normalized_center_of_labels(full_processed_label_img1)
        tag_center["image2_processed"] = segmentation_tool.get_normalized_center_of_labels(full_processed_label_img2)

        start_merging_process(tag_center)
      
        #Ending the program and saving all the necessary info
        data_project = {
            "scale_factor": str(scale_factor),
            "rotation_factor": str(angles[0]),
            "img1_processing": f"{tags_info['image1']['upperLayersToRemove']}, {tags_info['image1']['lowerLayersToRemove']}",
            "img2_processing": f"{tags_info['image2']['upperLayersToRemove']}, {tags_info['image2']['lowerLayersToRemove']}",
            "center_bead1": f"{center1[0]}, {center1[1]}",
            "center_bead2": f"{center2[0]}, {center2[1]}",
            "radius_bead2": str(radius2),
            "cropping_value": str(tags_info["croppingValue"]),
            "phalo_tag": str(tags_info["phaloTag"]),
            "last_merging_run": "Automatic merging"       
        }


        utils.save_project_info(data_project, project_path)
        utils.modify_running_process("Done Merging auto", json_path)

    except Exception as e:
        utils.modify_running_process("Error while running the auto merging process", json_path)
        print(e)