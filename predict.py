
# Imports here
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


# TODO: Write a function that loads a checkpoint and rebuilds the model
"""Loading the Checkpoint"""
"""Note for imports this requires torchvision models module"""

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath, weights_only=False)

    hidden_units = checkpoint['hidden_units']
    output_units = max(int(0.2 * hidden_units), 102)
    print("Output Units:", output_units)

    model = models.vgg16(pretrained=True)
    
    class Classifier(nn.Module):
      def __init__(self):
        super().__init__()

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


    model.classifier = Classifier()

    
    model.load_state_dict(checkpoint['model_state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']

    #idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    #Follow up: there is a requirement to save the optimizer state but there is no use for it follow up on this
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()

    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    size = 256, 256

    with Image.open(image) as im:


        #print('flowers/train/1/image_06734.jpg',im)

        """im.thumbnail(size)

        width, height = im.size

        if width < height:
            im = im.resize((256, int(height * (256 / width))))
        else:
            im = im.resize((int(width * (256 / height)), 256))

        left = (width - 224) / 2
        top = (height - 224) / 2
        right = (width + 224) / 2
        bottom = (height + 224) / 2

        cropped_im = im.crop((left, top, right, bottom))

        np_image = np.array(cropped_im)"""

        # Method 1: we are looping through this image 3 times because there is 3 color channels
        # We looking at a on specific cell and looking at changes to the color channel
        # np_image[1, 1, 0] = 13
        # (13 - 0.485) / 0.229 = 54.65
        #for i in range(3):
        #    print("color channel:", np_image[1, 1, 0])
        #    np_image[:, :, i] = (np_image[:, :, i] - mean[i]) / std[i]
        #    print("After color channel:", np_image[1, 1, 0])"""

        #Method 2:
        #print("color channel:", np_image[1, 1, 0])
        #np_image = np_image - mean / std
        #print("After color channel:", np_image[1, 1, 0])

        #we are converting this back to torch tensor because the model is expecting a torch tensor as well as transposing the the color channel
        #np_image = torch.from_numpy(np_image)
        #tensor_image = np_image.permute(2, 0, 1)
        #channels of images are typically encoded as integers 0-255, but the model expected floats 0-1.
        #tensor_image = tensor_image.float()/255
        #print("Color channel", np_image[1,1,1])

        #Method 3: using torchvision transforms to normalize the image color channels
        #we are converting this back to torch tensor because the model is expecting a torch tensor as well as transposing the the color channel
        #np_image = torch.from_numpy(np_image)
        #tensor_image = np_image.permute(2, 0, 1)
        #channels of images are typically encoded as integers 0-255, but the model expected floats 0-1.
        #tensor_image = tensor_image.float()/255
        #transform = transforms.Normalize(mean, std)
        #tensor_image = transform(tensor_image.float())
        #print("After Normalize transformation: ", np_image[0,1,1])

        data_transforms = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        tensor_image = data_transforms(im)


    return tensor_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    input = process_image(image_path)# takes in an image from file and then feeds into the process_image function to convert to NP array values

    #https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    input = input.type(torch.FloatTensor).unsqueeze(0)

    #commenting out as model is already set to eval using the load_checkpoint() function
    #model.eval()

    device = "cpu" 

    if in_arg.gpu is True:
        #sets default accelerator
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    print("Device:", device)
    model.to(device)
    with torch.no_grad():
      input = input.to(device)
      logps = model.forward(input)

    print("Log probabilities (logps):", logps) # Inspect logps

    ps = torch.exp(logps)
    print("Probabilities (ps):", ps) # Inspect ps
    top_p, top_class = ps.topk(topk) # remember ps.topk returns the k highest values in the form of two tuples top_p is the probability values, top_class class indeces
    print("Top_P Before:", top_p)
    print("Top_Class Before:", top_class)

    """
    top_p.detach(): This creates a new tensor that is detached from the current computation graph, preventing gradient calculation. This is good practice for inference.
.cpu(): This moves the tensor from the GPU (if it was on one) to the CPU.
.numpy(): This converts the PyTorch tensor to a NumPy array.
.tolist(): This converts the NumPy array to a Python list. Since the NumPy array is 2D (shape (1, 5)), this results in a list of lists (e.g., [[0.9999, 0.00010091, ...]]).
[0]: This accesses the first element of the list of lists, which is the inner list containing the probabilities.
    """
    top_p = top_p.detach().cpu().numpy().tolist()[0]
    top_c = top_class.detach().cpu().numpy().tolist()[0]
    print("Top_P After:", top_p)
    #EXPERIMENTAL
    # We need to m
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    top_c = [idx_to_class[prediction] for prediction in top_c]

    #flower = [cat_to_name[idx_to_class[prediction]] for prediction in top_c]

    return top_p, top_c


def get_input_args():
  
   
    parser = argparse.ArgumentParser(description='Program accepts Directory path as argument otherwise current folder is set as default')

    # TODO 0: Doing a default argument of os.getcwd() might not work because you need to define a default folder name which can be different
    # or we can just assume that the directoy retrieved in os.getcwd has the train and test folders as well as the cat_to_name.json file
    #parser.add_argument('--dir', nargs = '?', default = os.getcwd() , help='Image Folder as --dir with default value "pet_images"')
    parser.add_argument('dir', nargs = '*', default = [os.getcwd()+'/IMG_20250703_065528 (1) copy.jpg', os.getcwd()+'/checkpoint.pth'], help='Path to Image Dataset Folder otherwise default is current working directory') #for now i will set nargs to '* as i don't know if it's fixed we need to accept 3 arguments 2 positional and maybe one optional'
    #parser.add_argument('checkpoint', default= os.getcwd()+'/checkpoint.pth', help='path to checkpoint for training model') #this is experimental for now trying to see if i should have a positional argument or catch the nargs
    parser.add_argument('--top_k', type=int, default= 5)
    parser.add_argument('--category_names', default= os.getcwd()+"/cat_to_name.json")    
    parser.add_argument('--gpu', action='store_true')
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

#get commandline arguments
in_arg = get_input_args()

with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

print(len(cat_to_name))

paths = '"'+os.getcwd()+'/IMG_20250703_065528 (1) copy.jpg"' + ' "checkpoint.pth"'
print("String:", paths)
print('Num Args:', type(in_arg.dir))
print('Args:', in_arg.dir[0])
print('Args2:', in_arg.dir[1])
print('Top_k:', type(in_arg.top_k))

# TODO: Display an image along with the top 5 classes
image_path = in_arg.dir[0]

#loading positional argument for model checkpoint
checkpoint = in_arg.dir[1]

model = load_checkpoint(checkpoint)

top_p, top_class = predict(image_path, model, in_arg.top_k)
print("top_p: ", top_p)
print("top_class", top_class)
image = process_image(image_path)


fig, (ax1, ax2) = plt.subplots(figsize=(6,9), nrows=2)
imshow(image, ax=ax1) # Display using imshow
ax1.axis('off')
ax1.set_title("image")
print("Top_p Type:", type(top_p))

probs = np.array(top_p).squeeze()
classes = np.array(top_class).squeeze() #changed variable name to classes to avoid conflict
ax2.barh(np.arange(in_arg.top_k), probs) # Display top 5
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(in_arg.top_k))
ax2.set_yticklabels([cat_to_name[str(c)] for c in classes], size='small') # changed variable name here as well
#ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()
plt.show()