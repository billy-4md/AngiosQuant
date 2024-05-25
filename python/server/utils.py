import os
import json
import re
import numpy as np
import czifile
import imageio

from xml.etree import ElementTree as ET
from csbdeep.utils import normalize
from scipy.ndimage import median_filter, gaussian_filter


"""-------------CZI IMAGES FUNCTIONS----------"""
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
 
def isolate_and_normalize_channel(TAG, image_name, tags_info):
    image, dic_dim, channel_dict, axes = open_image(tags_info[image_name]['path'])
    channel_name = channel_dict[TAG]
    channel_index = get_channel_index(channel_name, channel_dict)
    if channel_index < 0 or channel_index >= dic_dim['C']:
        raise ValueError("Channel index out of range.")

    image_czi_reduite, axes_restants = czi_slicer(image, axes, indexes={'C': channel_index})

    return image_czi_reduite.astype(np.uint16)



"""-------------GENERAL FUNCTIONS----------"""
def create_full_image(image_name, tags_info, project_path):
    full_img = []
    cropped_value = tags_info["croppingValue"]
    up_slice_to_remove1 = tags_info["image1"]["upperLayersToRemove"]
    down_slice_to_remove1 = tags_info["image1"]["lowerLayersToRemove"]
    up_slice_to_remove2 = tags_info["image2"]["upperLayersToRemove"]
    down_slice_to_remove2 = tags_info["image2"]["lowerLayersToRemove"]
    phaloidin_tag = tags_info["phaloTag"]

    for tag_label, wavelength in tags_info[image_name]['tags'].items():
        if tag_label != f"TAG{phaloidin_tag}":
            tag_img = isolate_and_normalize_channel(wavelength, image_name, tags_info)
            if image_name == "image1":
                tag_img_cropped = tag_img[up_slice_to_remove1:-down_slice_to_remove1, cropped_value:-cropped_value, cropped_value:-cropped_value]
            elif image_name == "image2":
                tag_img_cropped = tag_img[up_slice_to_remove2:-down_slice_to_remove2, cropped_value:-cropped_value, cropped_value:-cropped_value]
            
            tag_img_cropped = normalize(tag_img_cropped, 1, 99.8, axis=(0,1,2))
            tag_img_cropped = median_filter(tag_img_cropped, size=3)
            gaussian_slice = gaussian_filter(tag_img_cropped, sigma=1)

            full_img.append(gaussian_slice)

    complete_img = np.maximum.reduce(full_img)
    save_image_function(complete_img, f"full_{image_name}.tif", project_path)

def modify_running_process(message, json_path):
    json_file_path = os.path.join(json_path, 'settings_data.json')
    with open(json_file_path, 'r') as file:
        data = json.load(file) 

    data["process"] = message

    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)  

def save_image_function(image, image_name, project_path):
    imageio.volwrite(os.path.join(project_path, image_name), image)

def stringify_keys(data):
    """
    Convertit récursivement toutes les clés de dictionnaire en chaînes de caractères.
    Gère également les listes contenant des dictionnaires.
    """
    if isinstance(data, dict):
        return {str(key): stringify_keys(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [stringify_keys(item) for item in data]
    else:
        return data

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convertit les arrays numpy en listes
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}  # Appel récursif pour les dictionnaires
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]  # Appel récursif pour les listes
    else:
        return obj

def prepare_data_for_json(data):
    if isinstance(data, dict):
        return {k: prepare_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [prepare_data_for_json(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data
       
def save_project_info(data_project, project_path):
    #Save project info in a json file
    file_path = os.path.join(project_path, "project_info.json")
    data_project_prepared = prepare_data_for_json(data_project)
    with open(file_path, 'w') as file:
        json.dump(data_project_prepared, file, indent=4)

def save_tag_centers(tag_center, project_path):
    #Save all centers in a json file
    file_path = os.path.join(project_path, "tag_centers.json")
    tag_center_str_keys = stringify_keys(tag_center)
    tag_center_str_keys_arr = convert_numpy(tag_center_str_keys)
    with open(file_path, 'w') as file:
        json.dump(tag_center_str_keys_arr, file, indent=4)  

def clean_project(project_path):
    data_project = {
        "error": "Error while running the auto process"        
    }

    file_path = os.path.join(project_path, "project_info.json")
    with open(file_path, 'w') as file:
        json.dump(data_project, file, indent=4)

