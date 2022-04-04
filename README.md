# HandsfreeMLWebApp

## Overview
This is a simple local webapp developed with Flask and HTML integrated with OpenCV, Tensorflow and Mediapipe. Its primary functionality is to detect the ASL alphabet hand sign.

## Set up
1. Clone this repo
2. To make sure your machine has all the necessary dependencies required:
  - Windows 
    1. cd to the directory which contains the `requirements.txt` file
    2. In terminal, run `pip install -r requirements.txt`
    3. After every dependencies is installed, run `python app.py`
  - Conda
    1. cd to the directory which contains the `requirements.txt` file
    2. In Conda terminal, run `conda install -file requirements.txt`
    3. After every dependencies is installed, run `python app.py`
    
## At runtime
1. Allow camera access to enable image being feed from webcam to backend process
2. `Categorized as` referred to the model prediction to the handsign you are current showing
3. `Confidence` referred to the model confidence on current prediction
4. You can access to the resources of ASL Handsign in `resources` tab

## Bug report
[Contact us](mailto:umquartet02@gmail.com) if any bug is found
