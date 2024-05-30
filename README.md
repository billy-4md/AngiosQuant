# PopSeg Tool

## Introduction
This project was developed as part of a Master Thesis in Civil Engineering with a biomedical specialization at ULB (Université Libre de Bruxelles), under the guidance of Professor Olivier Debeir and Egor Zindy. The complete report for this project is available in the "PDF" folder. **PopSeg** is a tool designed for processing and analyzing microscopic cellular images, particularly within the scope of multiplex immunostaining experiments. It offers an intuitive user interface and various features that simplify the analysis process, mainly designed for researchers and professionals in cellular biology.

## Structure of the project
In the "master" branch, all the code of this project is available. And it is structures in 3 main folders:
- **node_modules**: contains all the files necessary to run and to package this application. This application was made with [Electron.js](https://github.com/electron/electron)
- **python**: Contains all the python code in order to run all the features available in the available such as merging images, segmentation or data extraction. It is also in this folder that all the information are stored and it is divided in different parts:
  - "bin": This folder contains all the python library in order to run the code. No python installation are necessary on the computer as everything is contained inside the application.
  - "server": It is divided into several parts as well:
    - "app.py": This script is the main python script as it will start the Flask server, used to create the bridge between the frontend and backend.
    - "project": All the information such as images, json,... of a specific project will be store there.
    - "json": All the json files that contain some information important for the application.
    - "napari_plugin_folder": Contains the plugin for Napari and Hough Transform, added in the Napari viewer, which is added to the application to visualize the different images.
    - "code_brouillon": Contains all the old version of the python scripts.
    - "auto_merge.py": Main python script to run the automatic merging process.
    - "semi_auto_merge.py": Main python script to run the semi-automatic merging process.
    - "find_pop.py": Main script to generate the excel file, containent the matching population for each cell whithin the image as well as segmented the phalloidin tag.
    - "utils.py" & "segmentations_tool.py": Contains some fonctions that are used in different scripts.
    - "napari_launcher.py": Main python script to start the Napari visualizer.
- **web**: Contains all the file for the interface of the application and is divided in different parts:
  - "html": Contain all the HTML file for the application.
  - "js": Contain all the JavaScript file for the application, creating the requests on the Flask server to run the python code.
  - "css": COntain all the file for the style of the application.

## Features
This application offers a variety of features organized into several key components of the application:

### Settings
- Configure application settings in the **Settings** tab.
- In this place, several settings must be set in order to run the different program proprely. 
- Ensure all configurations are saved before proceeding to other sections.

### Project Management
- Create and manage projects via the **Project** tab.
- Upload images directly into projects for processing.

### Semi-Automatic Control Point Setup
This part is necessary if you need to run the semi-automatic algorithm. It allows to select, manually using "control points", corresponding to cell that can be seen on both images. If this option is selected, the algorithm must be run twice. The first time, it will juste add the images processed on your project. After having those images, you can set the control points, and then re run the program. Previous control points that have been set will be automatically visible and can be modify.
- If "Semi-Automatic" mode is selected, specify control points on original images.
- Access control points setting through the project page by selecting images and using the "Set Control Points" option in the navigation bar.
- Transition to 2D view for precise control point placement and save settings with keyboard shortcuts.

### Running Programs
- Two main functionalities under "Run Program":
  - **Image Fusion:** Automatically or semi-automatically fuse images based on predefined settings.
  - **Excel Generation:** Produce Excel files that match different cell populations based on analysis.

### Additional Information
- Each project contains a "Show Project Info" button, providing insights into the values obtained from the program's run.
- Modify values in the generated Excel to experiment with different analysis configurations.

## Configuration Setup
### System Requirements
- **Operating System:** Windows 64-bit
- **RAM:** Minimum 32GB
- **Network:** Ensure that port 8000 is open for the Python Flask server.
- **Software Requirements:** Microsoft Excel must be installed to open generated Excel files.

## Installation Methods
### Method 1: Using the Executable

1. Download `SetUp.exe` from the releases section.
2. Execute the downloaded file to install the application.
   - During installation, you can choose to add a desktop shortcut.

### Method 2: Using Source Code
1. Download the code folder in the master branch
2. Install ["Node.js"](https://nodejs.org/en/download/package-manager)
3. Run: npm install
4. Run: npm start to laucnh the application

 

## Output
The tool can generate images and Excel files depending on the selected options:
- Open images in the Napari viewer or view Excel files directly from the project page.


