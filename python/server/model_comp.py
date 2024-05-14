import os
import sys
#os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import json
import imageio
import czifile
import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from xml.etree import ElementTree as ET
from csbdeep.utils import normalize
from scipy.ndimage import median_filter, center_of_mass, gaussian_filter, rotate
from skimage.transform import resize
from scipy.spatial.distance import cdist
from cellpose import models, io, utils, plot
import skimage
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from scipy.ndimage import binary_fill_holes


def open_image(image_path):
    with czifile.CziFile(image_path) as czi:
        image_czi = czi.asarray()
        dic_dim = dict_shape(czi)  
        axes = czi.axes  
        metadata_xml = czi.metadata()
        metadata = ET.fromstring(metadata_xml)
        channels_dict = channels_dict_def(metadata)

    return image_czi, dic_dim, channels_dict, axes

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

def get_channel_index(channel_name, channel_dict):
    count = 0
    for keys, values in channel_dict.items():
        if values == channel_name:
            return count
        else:
            count += 1
  
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
 
def isolate_and_normalize_channel(TAG, image_name):
    image, dic_dim, channel_dict, axes = open_image(tags_info[image_name]['path'])
    channel_name = channel_dict[TAG]
    channel_index = get_channel_index(channel_name, channel_dict)
    if channel_index < 0 or channel_index >= dic_dim['C']:
        raise ValueError("Channel index out of range.")

    image_czi_reduite, axes_restants = czi_slicer(image, axes, indexes={'C': channel_index})

    return image_czi_reduite.astype(np.uint16)

def create_full_image(image_name):
    full_img = []

    for tag_label, wavelength in tags_info[image_name]['tags'].items():
        tag_img = isolate_and_normalize_channel(wavelength, image_name)
        if image_name == "image1":
            tag_img_cropped = tag_img[up_slice_to_remove1:-down_slice_to_remove1, cropped_value:-cropped_value, cropped_value:-cropped_value]
        elif image_name == "image2":
            tag_img_cropped = tag_img[up_slice_to_remove2:-down_slice_to_remove2, cropped_value:-cropped_value, cropped_value:-cropped_value]
        
        tag_img_cropped = normalize(tag_img_cropped, 1, 99.8, axis=(0,1,2))
        tag_img_cropped = median_filter(tag_img_cropped, size=3)
        gaussian_slice = gaussian_filter(tag_img_cropped, sigma=1)
        full_img.append(gaussian_slice)

    return np.maximum.reduce(full_img)

def save_image_function(image, image_name):
    if save_images:
        imageio.volwrite(os.path.join("D:\\Users\\MFE\\Xavier\\Result_Rapport2", image_name), image)


"""-------------MERGING FUNCTIONS----------"""
def translate_image(image, y_offset, x_offset, value = True):
    if value:
        scaled_image = zoom(image, (scale_factor, scale_factor), order=0)
    else:
        scaled_image = image

    translated_image = np.zeros_like(scaled_image)
    
    for y in range(scaled_image.shape[0]):
        for x in range(scaled_image.shape[1]):
            new_y = y + y_offset
            new_x = x + x_offset
            
            # Vérifie si les nouvelles coordonnées sont dans les limites de l'image
            if 0 <= new_y < scaled_image.shape[0] and 0 <= new_x < scaled_image.shape[1]:
                translated_image[new_y, new_x] = scaled_image[y, x]
                    
    return translated_image

def merge_images(angle, axes=(1, 0)):
    img1 = imageio.volread("C:\\Users\\MFE\\Xavier\\Outil3.0\\python\\server\\project\\Processed\\image1_processed.tif")
    img2 = imageio.volread("C:\\Users\\MFE\\Xavier\\Outil3.0\\python\\server\\project\\Processed\\image2_processed.tif")

    z_dim, y_dim, x_dim = min(img1.shape[0], img2.shape[0]), img1.shape[1], img1.shape[2]
    merged_img_pos = np.zeros((z_dim, y_dim, x_dim), dtype=img1.dtype)
    merged_img_neg = np.zeros((z_dim, y_dim, x_dim), dtype=img1.dtype)

    if img1.shape[0] < img2.shape[0]:
        diff_slice = img2.shape[0] - img1.shape[0]
        img2 = img2[diff_slice//2:-diff_slice//2, :, :]
    elif img1.shape[0] > img2.shape[0]:
        diff_slice = img1.shape[0] - img2.shape[0]
        img1 = img1[diff_slice//2:-diff_slice//2, :, :]


    for z in range(z_dim):
        slice_img1 = img1[z, :, :]
        slice_img2 = img2[z, :, :]

        rotated_img_pos = rotate(slice_img1, angle[0], reshape=False, axes=axes)
        rotated_img_neg = rotate(slice_img1, -angle[0], reshape=False, axes=axes)
        slice_img1_pos = resize(rotated_img_pos, (y_dim, x_dim), order=0, mode='edge', anti_aliasing=False)
        slice_img1_neg = resize(rotated_img_neg, (y_dim, x_dim), order=0, mode='edge', anti_aliasing=False)

        merged_img_pos[z, :, :] = np.maximum(slice_img1_pos, slice_img2)
        merged_img_neg[z, :, :] = np.maximum(slice_img1_neg, slice_img2)

    save_image_function(merged_img_pos, 'img_merged_pos.tif')
    save_image_function(merged_img_neg, 'img_merged_neg.tif')
    print("MERGED HERE DONE")

def apply_transfo_img(value):
    img1 = imageio.volread("C:\\Users\\MFE\\Xavier\\Outil3.0\\python\\server\\project\\Processed\\image1_processed.tif")
    img2 = imageio.volread("C:\\Users\\MFE\\Xavier\\Outil3.0\\python\\server\\project\\Processed\\image2_processed.tif")
    
    start_segmentation(img1, f'image1_processed', 4500, 1500)
    start_segmentation(img2, f'image2_processed', 4500, 1500)
    keep_closest_point_between_img()

def keep_closest_point_between_img(z_weight=0.9):
    center_img1 = tag_center["image1_processed"]
    center_img2 = tag_center["image2_processed"]
    
    pt_cloud1 = []
    pt_cloud2 = []

    for (label1, center1), (label2, center2) in zip(center_img1.items(), center_img2.items()):
        weighted_center1 = np.array([center1[0], center1[1], center1[2] * z_weight])
        weighted_center2 = np.array([center2[0], center2[1], center2[2] * z_weight])
        pt_cloud1.append(weighted_center1)
        pt_cloud2.append(weighted_center2)

    print(len(pt_cloud1))
    print(len(pt_cloud2))
    #angles = start_icp(pt_cloud1, pt_cloud2, 0.05)
    angles = start_icp(pt_cloud1, pt_cloud2, 0.1)
    #angles = start_icp(pt_cloud1, pt_cloud2, 0.25)
    #angles = start_icp(pt_cloud1, pt_cloud2, 0.5)
    #angles = start_icp(pt_cloud1, pt_cloud2, 0.75)
    #angles=[-5.5, 0]
    merge_images(angles)

def prepare_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def compute_fpfh_feature(pcd):
    radius_normal = 0.1
    radius_feature = 0.25
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def perform_icp_point_to_plane(source_pcd, target_pcd, threshold):
    trans_init = np.eye(4)
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def perform_icp_ransac(source_pcd, target_pcd, source_fpfh, target_fpfh, distance_threshold):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # Four points to determine a plane
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))
    return result

def perform_basic_icp(source_pcd, target_pcd, threshold):
    trans_init = np.eye(4)  

    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def start_icp(points1, points2, threshold):
    pcd1 = prepare_point_cloud(points1)
    pcd2 = prepare_point_cloud(points2)
    fpfh1 = compute_fpfh_feature(pcd1)
    fpfh2 = compute_fpfh_feature(pcd2)

    print(f"--------------------Value : {threshold}---------------")

    result_basic = perform_basic_icp(pcd1, pcd2, threshold)
    print("Transformation matrix Basic:")
    print(result_basic.transformation)
    rotation_matrix = np.array(result_basic.transformation[:3, :3], dtype=np.float64, copy=True)
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True) 
    print(euler_angles)

    # result_ransac = perform_icp_ransac(pcd1, pcd2, fpfh1, fpfh2, threshold)
    # print("Transformation matrix Ransac:")
    # print(result_ransac.transformation)
    # rotation_matrix = np.array(result_ransac.transformation[:3, :3], dtype=np.float64, copy=True)
    # rotation = R.from_matrix(rotation_matrix)
    # euler_angles = rotation.as_euler('xyz', degrees=True) 
    # print(euler_angles)

    # result_plane = perform_icp_point_to_plane(pcd1, pcd2, threshold)
    # print("Transformation matrix Point-to-Plane:")
    # print(result_plane.transformation)
    # rotation_matrix = np.array(result_plane.transformation[:3, :3], dtype=np.float64, copy=True)
    # rotation = R.from_matrix(rotation_matrix)
    # euler_angles = rotation.as_euler('xyz', degrees=True) 
    # print(euler_angles)

    return euler_angles


"""-------------SEGMENTATION PROCESS FUNCTIONS----------"""
def do_segmentation(isolate_image, SURFACE_THRESHOLD_SUP, SURFACE_THRESHOLD_INF, diam_value):
    labels_all_slices = []
    data = []

    for z in range(isolate_image.shape[0]):
        print(f"Processing slice {z}")
        img_slice = isolate_image[z, :, :]
        # Normalize and filter the image slice
        #img_slice = normalize(img_slice, 1, 99.8, axis=(0,1))
        #filtered_img_slice = median_filter(img_slice, size=3)
        #filtered_img_slice = ((filtered_img_slice - np.min(filtered_img_slice)) / (np.max(filtered_img_slice) - np.min(filtered_img_slice)) * 255).astype(np.uint8)

        filtered_img_slice = img_slice
        # Evaluate the model with CellPose
        masks, flows, styles, diams = model.eval(filtered_img_slice, diameter=diam_value, channels=[0,0])
        #io.masks_flows_to_seg(filtered_img_slice, masks, flows, f"C:\\Users\\MFE\\Xavier\\Outil3.0\\python\\server\\project\\ModelComp\\save_result{z}", [0,0])

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

    return labels_all_slices, data
        
def reassign_labels(labels_all_slices, df):
    new_labels_all_slices = []
    max_label = 0  # Pour garder une trace du dernier label utilisé
    seuil = np.mean(labels_all_slices[0].shape) * 0.05

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

def start_segmentation(image, image_name, area_sup, area_inf):
    labels_all_slices, data = do_segmentation(image, SURFACE_THRESHOLD_SUP = area_sup, SURFACE_THRESHOLD_INF = area_inf)
    print(data)

    labels_all_slices = np.stack(labels_all_slices, axis=0)
    df = pd.DataFrame(data)

    labels = reassign_labels(labels_all_slices, df)
    save_image_function(labels, f'label_{image_name}.tif')

    tag_center[image_name] = get_normalized_center_of_labels(labels)

    return labels

def get_normalized_center_of_labels(label_image):
    unique_labels = np.unique(label_image)[1:]
    dims = np.array(label_image.shape)

    normalized_centers = {}

    for label in unique_labels:
        center = np.array(center_of_mass(label_image == label))
        normalized_center = center / dims
        normalized_centers[label] = normalized_center

    return normalized_centers

def launch_segmentation_process():
    global json_path, project_path
    global tag_center, tags_info, model
    global count_pop, cropped_value, up_slice_to_remove1, up_slice_to_remove2, down_slice_to_remove1, down_slice_to_remove2
    global save_images
    global circles1, circles2, scale_factor

    #Global variable
    resources_path = os.environ.get('FLASK_RESOURCES_PATH', os.getcwd())
    json_path = os.path.join(resources_path, 'python', 'server', 'json') 

    #Load json data
    json_file_path = os.path.join(json_path, 'settings_data_cellpose.json')
    with open(json_file_path, 'r') as f:
        tags_info = json.load(f)
    

    project_name = tags_info["project_name"]
    project_path = os.path.join(resources_path, 'python', 'server', 'project', project_name) 
    
    tag_center = {}
    count_pop = 1 
    cropped_value = tags_info["croppingValue"]
    up_slice_to_remove1 = tags_info["image1"]["upperLayersToRemove"]
    down_slice_to_remove1 = tags_info["image1"]["lowerLayersToRemove"]
    up_slice_to_remove2 = tags_info["image2"]["upperLayersToRemove"]
    down_slice_to_remove2 = tags_info["image2"]["lowerLayersToRemove"]
    save_images = tags_info["configuration"]["saveAllImages"]
    images = ["image1"]
    diam_value_list = [30, 60, 90]
    model_name_list = ["nuclei", "cyto", "cyto3"]
    
    #Configuration first
    if not os.path.exists(project_path):
        os.mkdir(project_path)


    #Images Processing
    for image in images:
        # full_image = create_full_image(image)
        # save_image_function(full_image, f"full_img_{image}.tif")
        full_image = imageio.volread("D:\\Users\\MFE\\Xavier\\Result_Rapport2\\img1_cropped.tif")

        for diam_value in diam_value_list:
            model = models.Cellpose(gpu=False, model_type="nuclei")
            print("Diam: ", diam_value)
            #3D
            masks, flows, styles, diams = model.eval(full_image, diameter=diam_value, channels=[0,0], do_3D =True)
            masks = masks.astype(np.uint16)
            save_image_function(masks, f"CellPose_pred_3D_nuclei_diam{diam_value}.tif")
            # #2D        
            # cellpose_pred, data = do_segmentation(full_image, 5000, 1500, diam_value)
            # df = pd.DataFrame(data)
            # labels = reassign_labels(cellpose_pred, df)
            # save_image_function(labels, f'CellPose_pred_2D_{"cyto"}_diam{diam_value}.tif')

           
    

    # model = models.Cellpose(gpu=False, model_type="cyto")
    # circles1=[[[790, 730, 370]]]
    # circles2=[[[775, 727, 420]]]
    # scale_factor = 1.15
    # print("Start merging")
    # apply_transfo_img("1")


launch_segmentation_process()