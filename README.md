# PopSeg Tool

## Introduction
This project was developed as part of a Master Thesis in Civil Engineering with a biomedical specialization at ULB (Université Libre de Bruxelles), under the guidance of Professor Olivier Debeir and Egor Zindy. The complete report for this project is available in the "PDF" folder. **PopSeg** is a tool designed for processing and analyzing microscopic cellular images, particularly within the scope of multiplex immunostaining experiments. It offers an intuitive user interface and various features that simplify the analysis process, mainly designed for researchers and professionals in cellular biology.

## Structure of the project
In the "master" branch, all the code of this project is available. And it is structures in 3 main folders:
- node_modules: contains all the files necessary to run and to package this application. This application was made with [Electron.js](https://github.com/electron/electron)
- python: Contains all the python code in order to run all the features available in the available such as merging images, segmentation or data extraction. It is also in this folder that all the images of the different project will be saved. It is the heart of the project as it contains the Flask server, which can create the bridge between the frontend and backend.
- web: Contains all the file for the interface of the application.

## Features
PopSeg offers a variety of features organized into several key components of the application:

### Settings
- Configure application settings in the **Settings** tab.
- Ensure all configurations are saved before proceeding to other sections.

### Project Management
- Create and manage projects via the **Project** tab.
- Upload images directly into projects for processing.

### Semi-Automatic Control Point Setup
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

## Getting Started
To get started with PopSeg, navigate to the `Settings` tab and configure the necessary settings. Create a new project in the `Project` tab and upload your images. Depending on your settings, you may need to set control points manually in the images for accurate processing.

## Semi-Automatic Mode Instructions
For projects running in semi-automatic mode, follow these steps to set control points:
1. Select your project and open the images.
2. Go to "Set Control Points" in the top navigation bar.
3. Switch to 2D view by clicking the cube icon on the bottom left of the interface.
4. Add points by clicking on the image with the "+" icon.
5. Save the points by pressing "s" on your keyboard; a confirmation message will appear.

## Output
The tool can generate images and Excel files depending on the selected options:
- Open images in the Napari viewer or view Excel files directly from the project page.

## Additional Help
For more detailed information and troubleshooting, refer to our [GitHub wiki](https://github.com/YourGitHubUsername/PopSeg/wiki) or issues section.

Thank you for choosing PopSeg for your image analysis needs!


