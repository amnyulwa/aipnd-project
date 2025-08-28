
# PROGRAMMER: Anesu S Mnyulwa 
# DATE CREATED:                                 
# REVISED DATE: 8/24/2025
# TODO  0:
#Train a new network on a data set with train.py

#Basic usage: python train.py data_directory
#Prints out training loss, validation loss, and validation accuracy as the network trains
#Options: * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory * Choose architecture: python train.py data_dir --arch "vgg13" * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 * Use GPU for training: python train.py data_dir --gpu

# Thoughts
# * We may need to figure out how to import list with class names the "cat_to_name.json" file


#Changes:
#Moving the model training back into the main function because I need to be able to save the model optimizer state we may need to improve this later

#TODO 0: Import required Libraries
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
import torch.nn.functional as F # This import doesn't seem to be used 
from torchvision import datasets, transforms, models, utils

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)



#def int_list(x):
  #  return [x]

def get_input_args():
  
   
    parser = argparse.ArgumentParser(description='Program accepts Directory path as argument otherwise current folder is set as default')

    # TODO 0: Doing a default argument of os.getcwd() might not work because you need to define a default folder name which can be different
    # or we can just assume that the directoy retrieved in os.getcwd has the train and test folders as well as the cat_to_name.json file
    #parser.add_argument('--dir', nargs = '?', default = os.getcwd() , help='Image Folder as --dir with default value "pet_images"')
    parser.add_argument('dir', nargs = '?', default = os.getcwd()+'/', help='Image Dataset Folder otherwise default is current working directory')        
    parser.add_argument('--save_dir', default='checkpoint.pth')
    parser.add_argument('--arch', default= "vgg")
    parser.add_argument('--learning_rate', type=float, default= 0.001)
    parser.add_argument('--hidden_units', type=int , default=512)
    parser.add_argument('--epochs', type=int, default=5)
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
class Classifier(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()

            
        output_units = max(int(0.2 * hidden_units), 102)
        print("Hidden units2:", hidden_units)
        print("Output Units:", output_units)
        # part-2 Neural Networks,  fully-connected or dense networks. Each unit in one layer is connected to each unit in the next layer. In fully-connected networks, the input to each layer must be a one-dimensional vector (which can be stacked into a 2D tensor as a batch of multiple examples)
        self.fc1 = nn.Linear(25088, hidden_units)            
        self.fc2 = nn.Linear(hidden_units, output_units)
        self.fc3 = nn.Linear(output_units, 102)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #print('Shape before {x.shape}')
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))        
        x = self.dropout(F.relu(self.fc2(x)))

        x = F.log_softmax(self.fc3(x), dim=1)

        return x

def model_test(testloader, model, device, criterion):
        # TODO: Do validation on the test set
    model.eval()
    #Note: make sure to initialize accuracy and test_loss to 0 other wise it will inherit values from the previous cell
    accuracy = 0
    test_loss = 0

    with torch.no_grad():

            for input, labels in testloader:

                    input, labels = input.to(device), labels.to(device)

                    logps = model.forward(input)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1) # remember ps.topk returns the k highest values in the form of two tuples top_p is the probability values, top_class class indeces
                    print(top_p)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy : {accuracy/len(testloader):.3f}")



def main():

    start_time = time()  

    #TODO 1: Command line arguments and variable mapping.
    #Please make sure to change this to CPU when you complete this project
    #device = "cuda:0"
    device = "mps"        
    
    in_arg = get_input_args()   
    print("Hidden Argsss PPP:",type(in_arg.hidden_units))

    # TODO 2: # setup data directories and Load data for Transforms
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

    #TODO 3: Label Mapping
    # Retrieve pet labels from the current working directory currently no arguments are setup        
    cat_to_name = get_label_mapping(data_dir)
    print(cat_to_name)

    
    #TODO 4: Building and training the classifier 

    #so far this method only accepts 3 types of models and if there is a need i can work on expanding the list or automate the process to show more models


    models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

    model = models[in_arg.arch] #accepts input from user for model architecture     
    
    print("Peew pew:", model)

    for param in model.parameters():
        param.requires_grad = False 

    
    #creating the model with input of 
    model.classifier = Classifier(in_arg.hidden_units)
   

    #TODO 5: Model training Epoch argument
    #It might be pointless to encapsulate in if arg since there is already a default argument defined
    epochs = in_arg.epochs
    print("Epochs:", in_arg.epochs)  

   
    if in_arg.gpu is True:
        device = torch.device("cuda:0")
        print("Device", device)
    
   
    print("The model again <<>>:", model)

    """
    Define Network parameters 

    """
    # Network train cho cho

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=in_arg.learning_rate) 

    model.to(device)
   
    steps = 0

    train_Ls, valid_Ls = [], []

    for epoch in range(epochs):
        running_loss = 0
        for input, labels in trainloader:
            steps +=1

            input, labels = input.to(device), labels.to(device)

            logps = model.forward(input)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            validation_loss = 0
            accuracy = 0

            model.eval()

            with torch.no_grad():

                for input, labels in validationloader:

                    input, labels = input.to(device), labels.to(device)
                    #Forward pass, get our log-probabilities
                    logps = model.forward(input)
                    batch_loss = criterion(logps, labels)
                    validation_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1) # remember ps.topk returns the k highest values in the form of two tuples top_p is the probability values, top_class class indeces
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            #calculating epoch's average training loss
            train_loss = running_loss / len(train_datasets)
            valid_loss = validation_loss / len(validationloader.dataset)


            #calculating average loss for each batch
            avg_Tbatch_loss = running_loss / steps
            avg_Vbatch_loss = validation_loss / steps

            epoch_loss = running_loss / len(trainloader.dataset)

            #Collect all test and training loss to enable graphing the data
            #this is saving each output from the training and validation loss
            train_Ls.append(epoch_loss)
            valid_Ls.append(valid_loss)

            #Print function is out putting training, valiation loss per epoch
            print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Training loss: {train_loss:.3f}.. "
                    f"Validation loss: {valid_loss:.3f}.. "
                    f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()# switching the model into train mode


    """
    Model Testing
    """ 
    model_test(testloader, model, device, criterion)
    """
    Save Model Checkpoint
    """
    print(train_datasets.class_to_idx)

    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {
                'pre_trained_model': models[in_arg.arch],                
                'hidden_units': in_arg.hidden_units,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epochs,
                'class_to_idx':model.class_to_idx}

    torch.save(checkpoint, in_arg.save_dir)      
  
    end_time = time()
    
   
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()



