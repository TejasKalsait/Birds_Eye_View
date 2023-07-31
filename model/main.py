#!/usr/bin/env python


import importlib
import os
import sys
import tqdm
import numpy as np
import cv2
import tensorflow as tf
import configargparse
from fastprogress import progress_bar

import utils


# parse parameters from config file or CLI
parser = configargparse.ArgParser()
parser.add("-c",    "--config", is_config_file=True, help="config file")
parser.add("-ip",   "--input-testing",          type=str, required=True, nargs="+", help="directory/directories of input samples for testing")
parser.add("-np",   "--max-samples-testing",    type=int, default=None,             help="maximum number of testing samples")
parser.add("-is",   "--image-shape",            type=int, required=True, nargs=2,   help="image dimensions (HxW) of inputs and labels for network")
parser.add("-ohi",  "--one-hot-palette-input",  type=str, required=True,            help="xml-file for one-hot-conversion of input images")
parser.add("-ohl",  "--one-hot-palette-label",  type=str, required=True,            help="xml-file for one-hot-conversion of label images")
parser.add("-m",    "--model",                  type=str, required=True,            help="Python file defining the neural network")
parser.add("-uh",   "--unetxst-homographies",   type=str, default=None,             help="Python file defining a list H of homographies to be used in uNetXST model")
parser.add("-mw",   "--model-weights",          type=str, required=True,            help="weights file of trained model")
parser.add("-pd",   "--prediction-dir",         type=str, required=True,            help="output directory for storing predictions of testing data")
conf, unknown = parser.parse_known_args()


# determine absolute filepaths
conf.input_testing          = [utils.abspath(path) for path in conf.input_testing]
conf.one_hot_palette_input  = utils.abspath(conf.one_hot_palette_input)
conf.one_hot_palette_label  = utils.abspath(conf.one_hot_palette_label)
conf.model                  = utils.abspath(conf.model)
conf.unetxst_homographies   = utils.abspath(conf.unetxst_homographies) if conf.unetxst_homographies is not None else conf.unetxst_homographies
conf.model_weights          = utils.abspath(conf.model_weights)
conf.prediction_dir         = utils.abspath(conf.prediction_dir)


# load network architecture module
architecture = utils.load_module(conf.model)


# get max_samples_testing samples
files_input = [utils.get_files_in_folder(folder) for folder in conf.input_testing]
_, idcs = utils.sample_list(files_input[0], n_samples=conf.max_samples_testing)
files_input = [np.take(f, idcs) for f in files_input]
n_inputs = len(conf.input_testing)
n_samples = len(files_input[0])
image_shape_original = utils.load_image(files_input[0][0]).shape[0:2]
print(f"Found {n_samples} samples")


# parse one-hot-conversion.xml
conf.one_hot_palette_input = utils.parse_convert_xml(conf.one_hot_palette_input)
conf.one_hot_palette_label = utils.parse_convert_xml(conf.one_hot_palette_label)
n_classes_input = len(conf.one_hot_palette_input)
n_classes_label = len(conf.one_hot_palette_label)


# build model
if conf.unetxst_homographies is not None:
  uNetXSTHomographies = utils.load_module(conf.unetxst_homographies)
  model = architecture.get_network((conf.image_shape[0], conf.image_shape[1], n_classes_input), n_classes_label, n_inputs=n_inputs, thetas=uNetXSTHomographies.H)
else:
  model = architecture.get_network((conf.image_shape[0], conf.image_shape[1], n_classes_input), n_classes_label)
model.load_weights(conf.model_weights)
print(f"Reloaded model from {conf.model_weights}")


# build data parsing function
def parse_sample(input_files):
    # parse and process input images
    inputs = []
    for inp in input_files:
        inp = utils.load_image_op(inp)
        inp = utils.resize_image_op(inp, image_shape_original, conf.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        inp = utils.one_hot_encode_image_op(inp, conf.one_hot_palette_input)
        inputs.append(inp)
    inputs = inputs[0] if n_inputs == 1 else tuple(inputs)
    return inputs


# create output directory
if not os.path.exists(conf.prediction_dir):
    os.makedirs(conf.prediction_dir)


# run predictions
print(f"Running predictions and writing to {conf.prediction_dir} ...")
for k in progress_bar(range(n_samples)):

    input_files = [files_input[i][k] for i in range(n_inputs)]

    # load sample
    inputs = parse_sample(input_files)

    # add batch dim
    if n_inputs > 1:
        inputs = [np.expand_dims(i, axis=0) for i in inputs]
    else:
        inputs = np.expand_dims(inputs, axis=0)

    # run prediction
    prediction = model.predict(inputs).squeeze()

    # convert to output image
    prediction = utils.one_hot_decode_image(prediction, conf.one_hot_palette_label)

    # write to disk
    output_file = os.path.join(conf.prediction_dir, os.path.basename(files_input[0][k]))
    
    cv2.imwrite(output_file, cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))
    
    if k == 50:
        print("Breaking because testing for", k, "images only")
        break
    
    #part_1_output = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
    
    
    
    #################### PART 2 ##########################\
        
    
print("Importing all necessary packages for part 2")
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import argparse

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

import xml.etree.ElementTree as ET
import os

from colorama import Fore
from fastprogress import progress_bar

print("All required packages imported")

# Define the CNN model
class CNN(nn.Module):
    
    def __init__(self):
        
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 0)
        self.fc1 = nn.Linear(111872, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, x):
        
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 111872)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    
transform = transforms.Compose([transforms.Resize((321, 201)),
                                transforms.ToTensor(),
                                transforms.Grayscale(),
                                transforms.Normalize((0.5), (0.5))])

input_dir = "predictions/"

p2_train_data = []
p2_labels = []

images_names = os.listdir(input_dir)
print("List of images", images_names)

for curr_name in images_names:
    
    image_name = input_dir + str(curr_name)
    
    temp_img = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
    temp_img = cv.resize(temp_img, (201, 321), interpolation = cv.INTER_LINEAR)
    p2_train_data.append(temp_img)
    p2_labels.append(1)

part_2_inputs = torch.tensor(np.array(p2_train_data), dtype = torch.float)
p2_labels = torch.tensor(np.array(p2_labels), dtype = torch.long)

part_2_dataset = TensorDataset(part_2_inputs, p2_labels)

test_loader = DataLoader(part_2_dataset, batch_size = 1, shuffle = False)
print("LENGTH OF PART 2 DATA", len(part_2_dataset))



# train_dataset = datasets.ImageFolder(root = '../data/content/Cam2BEV/data/2_F/val/bev+occlusion/', transform = transform)

# print("Length of train data is", Fore.RED, len(train_dataset), Fore.RESET)

# test_dataset = datasets.ImageFolder(root = 'test', transform = transform)

# train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True)
# test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = True)

#print("Number of items in train loader is", Fore.RED, len(train_loader), Fore.RESET, "and first value is", train_loader[0],"and type of items in tuple are", Fore.RED, train_loader[0][0].shape, train_loader[0][1].shape, Fore.RESET)

model = CNN()
#print(model)

model.load_state_dict(torch.load("part_2_model.pth"))

#reduced_image = cv2.resize(part_1_output, (201, 321), interpolation = cv.INTER_LINEAR)

arrow_dir = "lane_change/"

count = 0
    
for x, y in progress_bar(test_loader):
    
    count += 1

    inp_img, _ = x, y
    
    outputs = model(inp_img)

    which_direction = torch.argmax(outputs)
    
    which_direction = which_direction.detach().numpy()


    print("Direction is", Fore.RED, which_direction, Fore.RESET)
    
    #print(inp_img.detach().numpy().shape)
    
    if which_direction == 0:

        arrow_image = cv2.arrowedLine(inp_img.detach().numpy().reshape(321, 201, 1), (160, 100), (160, 50), (255, 0, 0), 3) 
        
    elif which_direction == 1:
        
        arrow_image = cv2.arrowedLine(inp_img.detach().numpy().reshape(321, 201, 1), (160, 100), (190, 100), (255, 0, 0), 3) 
    
    else:
        arrow_image = cv2.arrowedLine(inp_img.detach().numpy().reshape(321, 201, 1), (160, 100), (160, 150), (255, 0, 0), 3) 
    
    cv2.imwrite(arrow_dir + str(count) + ".png", arrow_image)

print("Finished Predicting")