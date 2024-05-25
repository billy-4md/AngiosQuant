import os
import sys 
import shutil
import json
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import subprocess
import threading
import multiprocessing

from napari_launcher import launch_napari_def
from auto_merge import main_auto_merge
from semi_auto_merge import main_semi_auto_merge
from find_pop import main_find_pop



app = Flask(__name__)
CORS(app, origins='*', support_credentials=True)




resources_path = os.environ.get('FLASK_RESOURCES_PATH', os.getcwd())
PROJECT_FOLDER = os.path.join(resources_path, 'python', 'server', 'project')
JSON_FOLDER = os.path.join(resources_path, 'python', 'server', 'json')
app.config['PROJECT_FOLDER'] = PROJECT_FOLDER
app.config['JSON_FOLDER'] = JSON_FOLDER


#Route to upload new images and add them to the folder "upload"
@app.route('/upload_image/<project_name>', methods=['POST'])
def upload_image(project_name):
    if 'file' not in request.files:
        return jsonify(success=False, message='No file provided')
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, message='No selected file')
    
    if file:
        filename = os.path.join(app.config['PROJECT_FOLDER'], project_name, file.filename)
        file.save(filename)
        return jsonify(success=True, image_url=filename)

#Allow to list all the images that are present in the "upload" folder
@app.route('/images_list/<project_name>')
def list_images(project_name):
    project_folder = os.path.join(app.config['PROJECT_FOLDER'], project_name)

    if not os.path.exists(project_folder):
        return jsonify({"error": "Project not found"}), 404

    files = os.listdir(project_folder)
    images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.czi', '.xlsx'))] 

    images_paths = [f"/image/{project_name}/{image}" for image in images]

    project_info = {}
    file_path = os.path.join(app.config['PROJECT_FOLDER'], project_name, "project_info.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                project_info = json.load(f)
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to read project info, the data might be corrupt"}), 500

    return jsonify(images=images_paths, project_info=project_info)

@app.route('/update_project_info/<project_name>', methods=['POST'])
def update_project_info(project_name):
    data = request.json
    project_folder = os.path.join(app.config['PROJECT_FOLDER'], project_name, "project_info.json")
    with open(project_folder, 'w') as file:
        json.dump(data, file, indent=4)
    return jsonify({"message": "Project info updated successfully"})


@app.route('/get_segmentation/<project_name>')
def get_segmentation(project_name):
    image_name = request.args.get('image_name')
    if not image_name:
        return jsonify(success=False, message="No image name provided"), 400

    segmentation_folder = os.path.join(app.config['PROJECT_FOLDER'], project_name, 'segmentation')
    segmentation_folder_img = os.path.join(segmentation_folder, image_name)

    if not os.path.exists(segmentation_folder):
        os.mkdir(segmentation_folder)

    if not os.path.exists(segmentation_folder_img):
        os.mkdir(segmentation_folder_img)
        return jsonify(success=True, message="Segmentation folder created")

    try:
        segmentation_files = [f for f in os.listdir(segmentation_folder_img) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.czi'))]
        return jsonify(success=True, segmentation_masks=segmentation_files)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

#Route to delete images from the upload floder
@app.route('/delete_image/<project_name>', methods=['DELETE'])
def delete_image(project_name):
    image_name = request.args.get('image_name') 
    if not image_name:
        return jsonify(success=False, message='No image name provided')

    image_path = os.path.join(app.config['PROJECT_FOLDER'], project_name, image_name)
    image_path_segmentation = os.path.join(app.config['PROJECT_FOLDER'], project_name, "segmentation", image_name)
    
    if os.path.isfile(image_path):
        try:
            os.remove(image_path)
            shutil.rmtree(image_path_segmentation)
            return jsonify(success=True, message='Image deleted successfully')
        except Exception as e:
            return jsonify(success=False, message=str(e))
    else:
        return jsonify(success=False, message='Image not found')

#Route to delete project floder
@app.route('/delete_project/<project_name>', methods=['DELETE'])
def delete_project(project_name): 

    project_path = os.path.join(app.config['PROJECT_FOLDER'], project_name)
    
    if os.path.isdir(project_path):
        try:
            shutil.rmtree(project_path)
            return jsonify(success=True, message='Project deleted successfully')
        except Exception as e:
            return jsonify(success=False, message=str(e))
    else:
        return jsonify(success=False, message='Project not found')



@app.route('/launch_napari/<project_name>')
def launch_napari(project_name):
    image_names = request.args.get('image_names')
    control_points = request.args.get('control_points', 'false').lower() == 'true'
    
    if not image_names:
        return jsonify(success=False, message="No image names provided")

    image_paths = []
    print(image_names)

    for name in image_names.split(','):
        parts = name.split('_')
        if parts[-1] == "seg":
            mask_name = '_'.join(parts[:-2])
            current_image_path = os.path.join(app.config['PROJECT_FOLDER'], project_name, "segmentation", parts[-2], mask_name)
            print(current_image_path)
        else:
            current_image_path = os.path.join(app.config['PROJECT_FOLDER'], project_name, name)

        image_paths.append(current_image_path)

    missing_images = [path for path in image_paths if not os.path.exists(path)]
    if missing_images:
        return jsonify(success=False, message=f"Some images were not found: {', '.join(missing_images)}")

    project_path = os.path.join(app.config['PROJECT_FOLDER'], project_name)
    multiprocessing.Process(target=launch_napari_def, args=(image_paths, project_path, control_points)).start()

    return jsonify(success=True, message="Napari launched for provided images")

@app.route('/merge_function/<project_name>')
def merge_function(project_name):
    settings_file = os.path.join(JSON_FOLDER, 'settings_data.json')
    
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings_data = json.load(f)
        
        auto_merging = settings_data["configuration"]["automaticMerging"]
        if auto_merging:
            multiprocessing.Process(target=main_auto_merge, args=(project_name,)).start()
            return jsonify(success=True, message="Merge automatic function launched")
        else:
            multiprocessing.Process(target=main_semi_auto_merge, args=(project_name,)).start()
            return jsonify(success=True, message="Merge semi-automatic function launched")   
    else:
        return jsonify({"error": "Settings not available"}),

@app.route('/generate_excel_function/<project_name>')
def generate_excel_function(project_name):
    multiprocessing.Process(target=main_find_pop, args=(project_name,)).start()
    return jsonify(success=True, message="Generated excel function launched")


    
@app.route('/open_excel_files/<project_name>', methods=['GET'])
def open_excel_files(project_name):
    excel_files = request.args.get('excel_files')
    if not excel_files:
        return jsonify(success=False, message="No Excel file names provided"), 400

    # Dossier où sont stockés les fichiers Excel
    project_folder = os.path.join(app.config['PROJECT_FOLDER'], project_name)
    
    try:
        for file_name in excel_files.split(','):
            file_path = os.path.join(project_folder, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.xlsx'):
                subprocess.Popen(['start', 'excel', file_path], shell=True)
            else:
                return jsonify(success=False, message=f"File not found: {file_name}"), 404

        return jsonify(success=True, message="Excel files are being opened")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500




@app.route('/create_project', methods=['POST'])
def create_project():
    data = request.json 
    project_name = data['projectName']
    
    project_path = os.path.join(PROJECT_FOLDER, project_name)
    segmentation_folder_path = os.path.join(project_path, "segmentation")

    if not os.path.exists(project_path):
        os.makedirs(project_path)
        os.makedirs(segmentation_folder_path)
        return jsonify(success= True, message= f'Le projet "{project_name}" a été créé avec succès.')
    else:
        return jsonify(success= False, message=f'Le projet "{project_name}" existe déjà.')

@app.route('/get_projects')
def get_projects():
    projects = [d for d in os.listdir(PROJECT_FOLDER) if os.path.isdir(os.path.join(PROJECT_FOLDER, d))]
    return jsonify(projects)


@app.route('/save_project', methods=['POST'])
def save_project():
    project_name = request.json.get('projectName')
    json_file_path = os.path.join(JSON_FOLDER, 'info.json')
    
    # Read existing data, update it, and write back to the JSON file
    with open(json_file_path, 'r+') as json_file:  # Open file in read/write mode
        try:
            data = json.load(json_file)  # Try to load existing data
        except json.JSONDecodeError:  # If the file is empty or invalid, start with an empty dictionary
            data = {}
        data['currentProject'] = project_name  # Update the project name
        
        json_file.seek(0)  # Go to the beginning of the file
        json.dump(data, json_file)  # Write the updated data
        json_file.truncate()  # Truncate the file in case new data is shorter than old
    
    # Respond to the client that the project was successfully saved
    return jsonify({'message': f'Project {project_name} saved successfully'})

@app.route('/save_image', methods=['POST'])
def save_image():
    image_name = request.json.get('imageName') 
    json_file_path = os.path.join(JSON_FOLDER, 'info.json')
    
    with open(json_file_path, 'r+') as json_file:  
        try:
            data = json.load(json_file) 
        except json.JSONDecodeError: 
            data = {}
        data['currentImage'] = image_name  
        
        json_file.seek(0)  
        json.dump(data, json_file)  
        json_file.truncate() 

    return jsonify({'message': f'Image {image_name} saved successfully'})


@app.route('/save_settings', methods=['POST'])
def save_settings():
    request_data = request.json
    print("Received data:", request_data)  # Pour le debug

    settings_data = {
        "image1": {
            "path": request_data.get('image1', {}).get('path', ''),
            "tags": request_data.get('image1', {}).get('tags', {}),
            "upperLayersToRemove": request_data.get('image1', {}).get('upperLayersToRemove', 0),
            "lowerLayersToRemove": request_data.get('image1', {}).get('lowerLayersToRemove', 0)
        },
        "image2": {
            "path": request_data.get('image2', {}).get('path', ''),
            "tags": request_data.get('image2', {}).get('tags', {}),
            "upperLayersToRemove": request_data.get('image2', {}).get('upperLayersToRemove', 0),
            "lowerLayersToRemove": request_data.get('image2', {}).get('lowerLayersToRemove', 0)
        },
        "configuration": {
            "saveAllImages": request_data.get('configuration', {}).get('saveAllImages', ''),
            "automaticMerging": request_data.get('configuration', {}).get('automaticMerging', ''),
            "useTagCenter": request_data.get('configuration', {}).get('useTagCenter', '')
        },
        "populations": request_data.get('populations', {}),
        "croppingValue": request_data.get('croppingvalue', {}),
        "project_name": request_data.get('saveDirectory', "Default_Name"),
        "phaloTag": request_data.get('phaloTag', "")
    }

    settings_file = os.path.join(JSON_FOLDER, 'settings_data.json')
    
    try:
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        with open(settings_file, 'w') as f:
            json.dump(settings_data, f, indent=4)
        return jsonify({"message": "Settings saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_settings', methods=['GET'])
def get_settings():
    settings_file = os.path.join(JSON_FOLDER, 'settings_data.json')
    
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings_data = json.load(f)
            return jsonify(settings_data), 200 
    else:
        return jsonify({"error": "Settings not available"}), 

@app.route('/start_segmentation')
def start_segmentation():
    multiprocessing.Process(target=launch_segmentation_process).start()
    return jsonify(success=True, message="Segmentation process started")



if __name__ == '__main__':
    app.run(debug=False, port=8000)



