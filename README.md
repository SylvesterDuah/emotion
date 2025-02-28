# Emotion Detector with OpenCV

##### This project implements a simple emotion detection system using Python and OpenCV. It leverages Haar cascades to detect faces and smiles in real-time through a webcam feed. Additionally, it includes a basic demonstration of detecting whether a person appears "awake" (eyes detected) or "sleepy" (eyes not detected) as a proxy for more complex emotion detection. While simplistic, this project serves as an introductory exploration into computer vision techniques for emotion recognition.

### Features

#### Real-time face detection
#### Smile detection within detected faces
#### Basic awake/sleepy state detection based on eye presence
#### Visual feedback through webcam feed with annotated text and bounding boxes


### Prerequisites

#### Before running this project, ensure you have the following installed:
#### Python 3.x
#### OpenCV-Python package

### Usage
#### 1. Clone this repository to your local machine.
#### 2. Ensure you have a working webcam connected to your computer.
#### 3. Run the script with Python: python emotion.py


#### The application will open a window showing the live feed from your webcam. Detected faces will be outlined with rectangles, and detected smiles will trigger the display of "Smiling" text below the face. The script also attempts to determine if a person is "Awake" or "Sleepy" based on eye detection and displays this state accordingly.

#### Press the ESC key to exit the application


### Limitations

#### This project uses Haar cascades for detection tasks, which, while effective for simple applications, have their limitations in accuracy and robustness, especially in varied lighting conditions or complex emotional state detection. For more advanced applications, consider using deep learning models trained on comprehensive datasets.

### Contributing
#### Contributions to enhance this project are welcome! Feel free to fork this repository, make your changes, and submit a pull request. Whether it's adding more sophisticated emotion detection algorithms, improving the existing logic, or fixing bugs, your input is valuable.# emotion
# emotion
