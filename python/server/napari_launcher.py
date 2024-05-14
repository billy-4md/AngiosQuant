import os
import json
import napari
from napari_plugin_folder.hough.napari_hough_circle_detector import CircleDetectorWidget
import numpy as np

resources_path = os.environ.get('FLASK_RESOURCES_PATH', os.getcwd())

def load_points(file_path):
        """Charge les points depuis un fichier CSV s'il existe."""
        if os.path.exists(file_path):
            return np.loadtxt(file_path, delimiter=',')
        else:
            return np.empty((0, 3))

# Afficher Napari
def launch_napari_def(image_paths, project_path, control_points):
    settings_file = os.path.join(resources_path, 'python', 'server', 'json', 'settings_napari.json')
    control_point1 = os.path.join(project_path, 'control_point1.txt')
    control_point2 = os.path.join(project_path, 'control_point2.txt')
    print(project_path)
    print(control_points)

    try:
        with open(settings_file, 'r') as f:
            settings_dico = json.load(f)

    except Exception as e:
        print(f"Error while loading the settings file: {e}")
        return
    
    try:
        if not os.path.exists(control_point1):
            with open(control_point1, 'w') as f:
                f.write('') 

        if not os.path.exists(control_point2):
            with open(control_point2, 'w') as f:
                f.write('') 

    except Exception as e:
        print(f"Error creating control points files: {e}")
        return

    viewer = napari.Viewer()
    points_data1 = load_points(control_point1)
    points_data2 = load_points(control_point2)

    for image_path in image_paths:
        if image_path.split(".")[-1].lower() == "czi":
            print("CZI")
            viewer.open(image_path, plugin='napari-czifile2')
        else:
            viewer.open(image_path)

    if "viewMode" in settings_dico and settings_dico["viewMode"] == "3D":
        viewer.dims.ndisplay = 3

    if "HoughPlugin" in settings_dico and settings_dico["HoughPlugin"] == "ON":
        try:
            viewer.window.add_dock_widget(CircleDetectorWidget(napari_viewer=viewer), area='right')
            print("Hough plugin activated successfully.")
        except ValueError as e:
            print(f"Hough plugin not found or failed to load: {e}")

    if "StarDistPlugin" in settings_dico and settings_dico["StarDistPlugin"] == "ON":
        try:
            viewer.window.add_plugin_dock_widget('stardist-napari', 'StarDist')
            print("StarDist plugin activated successfully.")
        except ValueError as e:
            print(f"StarDist plugin not found or failed to load: {e}")

    if control_points == True:
        try:
            points_layer1 = viewer.add_points(
                points_data1, name='Control Points Image 1', size=10, face_color='red'
            )
            points_layer2 = viewer.add_points(
                points_data2, name='Control Points Image 2', size=10, face_color='blue'
            )
        except ValueError as e:
            print(f"Error when loading the control points layers: {e}")

    

    # Fonction pour enregistrer les points de la première couche
    def save_points1():
        np.savetxt(control_point1, points_layer1.data, delimiter=',')
        print("Control Points saved for Image 1!")

    # Fonction pour enregistrer les points de la deuxième couche
    def save_points2():
        np.savetxt(control_point2, points_layer2.data, delimiter=',')
        print("Control Points saved for Image 2!")

    # Fonction pour afficher un message dans le viewer
    def show_message(message):
        """Affiche un message temporaire dans le viewer."""
        viewer.status = message  # Affiche un message dans la barre d'état
        viewer.text_overlay.text = message

    # Ajouter des boutons de sauvegarde dans le viewer
    @viewer.bind_key('s')
    def save_all(viewer):
        """Enregistre les points de contrôle."""
        save_points1()
        save_points2()
        show_message("Control points have been saved!")


    napari.run()
