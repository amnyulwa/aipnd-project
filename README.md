# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


**Load the Training data**

Method 1

!wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
!unlink flowers
!mkdir flowers && tar -xzf flower_data.tar.gz -C flowers

Method 2

!cp -r /data/ .


**Python Packages required** 

!pip install torch 
!pip install torchvision 
!pip install numpy
!pip install matplotlib

for use with the following imports

from torch import nn
from torch import optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils



**Commands** 
  
  **1. Training the Model**
  Train model commands can be run as it accepting default arguments make sure that the "data" file with training data is in the same file path as the train.py file. The command can be used like below without arguments 
  
  !python train.py 
  
  with arguments argumment that have the -- character are optional  --gpu option turns on model training using a GPU if your hardware supports it: 
  
  !python train.py /path/to/training/data/ --epochs 10 --learning_rate 0.001 --gpu 
  !python predict.py input checkpoint --category_names cat_to_name.json

  **2. Using model for predictions**

  Note before running the prediction a checkpoint.pth file is required to save the model state after training 

  Commands:

  Basic usage: python predict.py /path/to/image checkpoint
Options: * Return top 
K
K most likely classes: python predict.py input checkpoint --top_k 3 * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json * Use GPU for inference: python predict.py input checkpoint --gpu
