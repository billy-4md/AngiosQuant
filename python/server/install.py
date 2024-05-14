import subprocess

def install_napari_package():
    try:
        result = subprocess.run(["napari", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["python3", "-m", "pip", "install", "napari[pyqt5]"])
        else:
            print("Napari installed!")
    except Exception as e:
        print(f"Error while installing Napari: {e}")


def install_napari_stardist_plugin():
    try:
        result = subprocess.run(["pip", "show", "stardist-napari"], capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["python3", "-m", "pip", "install", "stardist-napari"])
        else:
            print("Stardist plugin installed!")

    except Exception as e:
        print(f"Error while installing Stardist plugin: {e}")


#Note: faire la même chose avec flask, flask_corr + pip install napari-czifile2

#Bien choisir une version de python supérieru à 3.8!!

#Main functions to start the installation:
install_napari_package()
install_napari_stardist_plugin()
