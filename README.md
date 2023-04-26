# ID Photo Editor

This repository includes the code for my personal learing project towards ID photo editor.
The target for this tool is to allow me to properly crop out my head for ID photo.
Though there are many online / mobile apps for this purpose, I would like take the 
opportunity to write some code myself and learn to utilize machine learning algorithms 
to make ID photo cut to best quality.

The project includes
* A glfw / imgui based python front end.
* Segmentation methods (YOLO)
* Trimap generation and manual editing tools
* Image Matting with Mattformer
* Croping tool with face mark location using Mediapipe

There is no new algorithm developed in this project.
It's puting existing models / algorithms together to form a personal tool.

### Usage
In the project folder, run `pip3 install -r requirements.txt` to install all the dependencies.
Run `python main.py` to start the GUI.
Fast walk-through about the functionalities is available [here](https://www.youtube.com/watch?v=SNrpKWFR_1I)