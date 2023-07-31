# Birds_Eye_View

## Examples of Output
![alt text](https://github.com/TejasKalsait/Birds_Eye_View/blob/main/images/bird_eye_sample.png?raw=true)

## Table of Contents
1. [Introduction](#introduction)
2. [Project Description](#project-description)
3. [Dataset - CamBeV](#dataset)
4. [Team Members](#team-members)
5. [Network Architecture](#network-architecture)
6. [Result](#result)
7. [Generated Outputs](#generated-outputs)
8. [Blind Spot Functionality](#blind-spot-functionality)
9. [Usage](#usage)
10. [Configurations](#configurations)
11. [Dependencies](#dependencies)
12. [Installation](#installation)
13. [Running the Project](#running-the-project)
14. [Contributing](#contributing)

## Introduction
Welcome to the Bird'sEye View Project using Deep Learning and the CamBeV dataset. This project aims to generate Bird's Eye View outputs from a given dataset of images captured from a camera mounted on a vehicle. The Bird's Eye View provides a top-down view of the surroundings, which can benefit various applications, such as autonomous driving, parking assistance, and obstacle detection.

## Project Description
The Bird'sEye View Project utilizes deep learning techniques to transform images captured by a vehicle-mounted camera into a top-down perspective. The generated Bird's Eye View output enhances the understanding of the vehicle's surroundings and aids in making better decisions during driving.
**The model learns the spacial features and transforms the image into a different plane by learning a Homography Matrix.**

## Dataset
The CamBeV dataset is a collection of images captured by a vehicle-mounted camera. It serves as the primary dataset for training our deep learning model. The dataset contains various scenes and scenarios, including urban environments, highways, and rural areas. By using CamBeV, our model can learn to generalize well across different driving conditions.

## Team Members
This project was executed by a team of 4 dedicated individuals, each contributing their expertise and skills to bring the Bird'sEye View Project to life. The team members are:
1. Tejas Pankaj Kalsait (Myself) - kalsaittejas10@gmail.com
2. Athindra Bandi - athindra@buffalo.edu
3. Lakshmi Ramaswamy Vishwanath - lramaswa@buffalo.edu
4. Samarth Gangwal - samarthg@buffalo.edu

## Network Architecture
#### 1) Bird's Eye View
![alt text](https://github.com/TejasKalsait/Birds_Eye_View/blob/main/images/model_part1.png?raw=true)
#### 1) Blind Spot Detection for Lane Change
![alt text](https://github.com/TejasKalsait/Birds_Eye_View/blob/main/images/model_part2.png?raw=true)

## Result
![alt text](https://github.com/TejasKalsait/Birds_Eye_View/blob/main/images/result_graph.png?raw=true)

## Generated Outputs
The generated Bird's Eye View outputs from our trained model are available in the 'predictions' directory. These output images showcase the transformation of the original camera-captured images into top-down views. The generated outputs demonstrate the effectiveness of our deep learning approach in creating Bird's Eye View representations.

## Blind Spot Functionality
As a significant extension to the project, we have added blind spot functionality using the generated Bird's Eye View output. The blind spot feature helps the driver identify potential blind spots around the vehicle and assists in avoiding collisions with nearby objects.
The outputs from the blind spot detection feature are located in/model/lane_change directory.

## Usage
The Bird's Eye View Project is intended for developers, researchers, or anyone interested in exploring the application of deep learning for transforming camera images into Bird's Eye Views. The project provides a foundation for building advanced driver assistance systems (ADAS) and autonomous vehicle technologies.

## Configurations
The project is highly configurable to suit various scenarios and hardware setups. All configuration options can be found in the 'config.yml' file present inside the /model directory. Users can modify parameters such as model architecture, training hyperparameters, and blind spot functionality settings to achieve the desired results.

## Dependencies
The project relies on the following dependencies:
- Python >= 3.8
- Pytorch
- Torchvision
- NumPy
- OpenCV
- Matplotlib
- FastProgress

## Installation
To set up the Bird's Eye View Project, follow these steps:

1. Clone the repository to your local machine:
```
git clone https://github.com/your-username/BirdsEyeViewProject.git
cd BirdsEyeViewProject
```

2. Create a virtual environment (optional but recommended):
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the project dependencies:
```
pip install -r requirements.txt
```

## Running the Project
To run the project, execute the 'main.py' script along with the available configuration options. Make sure the necessary dataset and pretrained model weights (if any) are in place before running.

```
python main.py --config model/config(choice).yml
```

## Contributing
We welcome contributions to the Bird'sEye View Project. If you find any issues or have ideas for improvements, please open an issue or submit a pull request.

We hope you find the Bird's Eye View Project helpful and insightful. If you have any questions or need assistance, don't hesitate to reach out to any of our team members mentioned above. Happy coding!
