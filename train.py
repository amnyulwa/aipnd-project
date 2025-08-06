# TODO  0:
#Train a new network on a data set with train.py

#Basic usage: python train.py data_directory
#Prints out training loss, validation loss, and validation accuracy as the network trains
#Options: * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory * Choose architecture: python train.py data_dir --arch "vgg13" * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 * Use GPU for training: python train.py data_dir --gpu

# Thoughts
# * We may need to figure out how to import list with class names the "cat_to_name.json" file

import argparse
import os
import json

from time import time, sleep

import torch, torchvision
from torch import nn
from torch import optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils

def get_input_args():
  
   
    parser = argparse.ArgumentParser(description='Program accepts Directory path as argument otherwise current folder is set as default')

    # TODO 2: Doing a default argument of os.getcwd() might not work because you need to define a default folder name which can be different
    # or we can just assume that the directoy retrieved in os.getcwd has the train and test folders as well as the cat_to_name.json file
    #parser.add_argument('--dir', nargs = '?', default = os.getcwd() , help='Image Folder as --dir with default value "pet_images"')
    parser.add_argument('dir', nargs = '?', default = os.getcwd()+'/', help='Image Dataset Folder otherwise default is current working directory')        
    parser.add_argument('--save_dir', default=os.getcwd())
    parser.add_argument('--arch', default= "vgg16")
    parser.add_argument('--learning_rate', type=float, default= 0.02)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

def get_label_mapping(dir_path):    

    with open(dir_path+'cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name
"""
def load_transform():
    data_dir = input
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.CenterCrop((224, 224)),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.CenterCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.CenterCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_datasets = datasets.ImageFolder(test_dir, transform=testing_transforms)

    print(train_datasets.class_to_idx)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # remember the batch size is the number of images we get in one iteration from the data loader and pass through our network, often called a batch. And shuffle=True tells it to shuffle the dataset every time we start going through the data loader again
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size=64)
    testloader = torch.utils.data.DataLoader(testing_datasets, batch_size=64)

    print(len(trainloader.dataset))
    print(len(trainloader))
"""

def main():
    
    start_time = time()    
    
    in_arg = get_input_args()

    # TODO 1: we need to define a function that checks the input argument to ensure the directory path is in the correct format,
    # and possibly you can expand to the directory path exists as well as the folder structure should have data>train>class, data>validation>class
    #check_command_line_arguments(in_arg)     
      
    print("Current Directory:{}, LR:{}, Epoch:{}".format(in_arg.dir, in_arg.learning_rate, in_arg.epochs))

    #setup data directories 
    # we may need to improve this such that the directory containing the data has a consistent name
    # We can update the folder name to be just data?
    data_dir = in_arg.dir     
    train_dir = data_dir + 'flowers' + '/train'
    valid_dir = data_dir + 'flowers' + '/valid'
    test_dir = data_dir + 'flowers' + '/test'
    
    """
    Load Transforms
    """
    train_transforms = transforms.Compose([transforms.CenterCrop((224, 224)),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.CenterCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.CenterCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_datasets = datasets.ImageFolder(test_dir, transform=testing_transforms)

    print(train_datasets.class_to_idx)
   
    # remember the batch size is the number of images we get in one iteration from the data loader and pass through our network, often called a batch. And shuffle=True tells it to shuffle the dataset every time we start going through the data loader again
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size=64)
    testloader = torch.utils.data.DataLoader(testing_datasets, batch_size=64)

    print(len(trainloader.dataset))
    print(len(trainloader)) 

    # Retrieve pet labels from the current working directory currently no arguments are setup        
    cat_to_name = get_label_mapping(in_arg.dir)
    print(cat_to_name)

    print("Learning rate:", in_arg.learning_rate)
    print("Epochs:", in_arg.epochs)
    if in_arg.arch:
        print("CNN Model Architecture:{}", in_arg.arch)            
    if in_arg.hidden_units:
        print("Hidden Units: {}", in_arg.hidden_units)  
    if in_arg.gpu is True:
        device = torch.device("cuda:0")
    
    print("Device", device)
    print("Pew pew")
    
    # Function that checks Pet Images in the results Dictionary using results    
    #check_creating_pet_image_labels(results)

    #TODO2: Load the data for transformation 
    
    # TODO 3: Define classify_images function within the file classiy_images.py
    # Once the classify_images function has been defined replace first 'None' 
    # in the function call with in_arg.dir and replace the last 'None' in the
    # function call with in_arg.arch  Once you have done the replacements your
    # function call should look like this: 
    #             classify_images(in_arg.dir, results, in_arg.arch)
    # Creates Classifier Labels with classifier function, Compares Labels, 
    # and adds these results to the results dictionary - results
    classify_images(in_arg.dir, results, in_arg.arch)

    # Function that checks Results Dictionary using results    
    check_classifying_images(results)    

    
    # TODO 4: Define adjust_results4_isadog function within the file adjust_results4_isadog.py
    # Once the adjust_results4_isadog function has been defined replace 'None' 
    # in the function call with in_arg.dogfile  Once you have done the 
    # replacements your function call should look like this: 
    #          adjust_results4_isadog(results, in_arg.dogfile)
    # Adjusts the results dictionary to determine if classifier correctly 
    # classified images as 'a dog' or 'not a dog'. This demonstrates if 
    # model can correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(results, None)
   
    # Function that checks Results Dictionary for is-a-dog adjustment using results
    check_classifying_labels_as_dogs(results)
    

    # TODO 5: Define calculates_results_stats function within the file calculates_results_stats.py
    # This function creates the results statistics dictionary that contains a
    # summary of the results statistics (this includes counts & percentages). This
    # dictionary is returned from the function call as the variable results_stats    
    # Calculates results of run and puts statistics in the Results Statistics
    # Dictionary - called results_stats
    results_stats = calculates_results_stats(results)

    # Function that checks Results Statistics Dictionary using results_stats
    check_calculating_results(results, results_stats)


    print_results(results, results_stats, in_arg.arch, True, True)
    
  
    end_time = time()
    
   
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()



