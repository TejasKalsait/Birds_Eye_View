print("Importing all necessary packages")
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

l_rate = 0.001
epochs = 5

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


# train_dataset = datasets.ImageFolder(root = '../data/content/Cam2BEV/data/2_F/val/bev+occlusion/', transform = transform)

# print("Length of train data is", Fore.RED, len(train_dataset), Fore.RESET)

# test_dataset = datasets.ImageFolder(root = 'test', transform = transform)

# train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True)
# test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = True)

label_dir = "../data/content/Cam2BEV/data/Annotations/"
data_dir = "../data/content/Cam2BEV/data/1_FRLR/val/bev+occlusion/real_data/"

xml_files = os.listdir(label_dir)

train_labels = []

train_data = []

print("Getting the data and labels. Might take around", Fore.RED, "15 seconds", Fore.RESET)

for file in xml_files:
    
    # Extract image path
    image_name = file[ : -4]
    image_file_name = data_dir + image_name + ".png"
    
    # Get the label and append
    tree = ET.parse(label_dir + file)
    root = tree.getroot()
    curr_label = root.find('object/name').text
    train_labels.append(int(curr_label))
    
    # Store image
    temp_img = cv.imread(image_file_name, cv.IMREAD_GRAYSCALE)
    temp_img = cv.resize(temp_img, (201, 321), interpolation = cv.INTER_LINEAR)
    print("Shape", temp_img.shape)
    train_data.append(temp_img)

train_labels = torch.tensor(np.array(train_labels), dtype = torch.long)
train_data = torch.tensor(np.array(train_data), dtype = torch.float)

dataset = TensorDataset(train_data, train_labels)

train_loader = DataLoader(dataset, batch_size = 1, shuffle = True)

print("Amount of training data got is", Fore.RED, len(train_loader), Fore.RESET)
#print("Number of items in train loader is", Fore.RED, len(train_loader), Fore.RESET, "and first value is", train_loader[0],"and type of items in tuple are", Fore.RED, train_loader[0][0].shape, train_loader[0][1].shape, Fore.RESET)

model = CNN()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = l_rate)

# Train the model

train_loss_record = []
train_accuracy_record = []

print("Starting the training")

for epoch in progress_bar(range(epochs)):
    
    running_loss = 0.0
    correct = 0
    total = len(train_labels)
    
    # show = True
    
    for x, y in progress_bar(train_loader):
        
        inputs, train_label = x, y
        
        # if show:
            
        #     print("Image looks like", inputs, inputs.shape)
        #     print("Train label looks like", train_label)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        #print("Outputs and train label are", Fore.RED, torch.argmax(outputs), Fore.RESET, "and", Fore.RED, train_label, Fore.RESET)
        loss = criterion(outputs, train_label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (torch.argmax(outputs) == train_label).sum().item()
        #print("Correct so far are", Fore.GREEN, correct, Fore.RESET)
        
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    print("Epoch", Fore.RED, epoch + 1, Fore.RESET, "done")     
    print("Training accuracy", Fore.RED, train_acc, Fore.RESET)
    print("Training Loss: ", Fore.RED, train_loss, Fore.RESET)
    
    train_loss_record.append(train_loss)
    train_accuracy_record.append(train_acc)
    
    
    
    # # Evaluate the model on the test dataset
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, test_labels = data
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += test_labels.size(0)
    #         correct += (predicted == test_labels).sum().item()

    # print("Testing accuracy", 100 * correct / total)
    
    
    torch.save(model.state_dict(), "../models/part_2_model.pth")

plt.plot(train_accuracy_record)
plt.plot(train_loss_record)
plt.title("Training record")
plt.show()