from model import ft_net, ft_net_swin, ft_net_swinv2, PCB, ft_net_dense
from torchvision import datasets, models, transforms
import numpy as np
import math
import os
import yaml
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import scipy.io
from scipy.spatial.distance import cosine
from tqdm import tqdm
import torch.nn as nn

version =  torch.__version__

use_swin = False
use_dense = True
epoch = "last"
linear_num = 512
batch_size = 256

use_gpu = torch.cuda.is_available()
gpu_ids = [0]
ms = []
ms.append(math.sqrt(float(1)))

if use_swin:
    h, w = 224, 224
    name = 'ft_net_swin'
    
else:
    h, w = 256, 128
    name = 'ft_net_dense'

interpolation_mode = transforms.InterpolationMode.BICUBIC

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=interpolation_mode),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

config_path = os.path.join('./model',name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'

if 'nclasses' in config: # tp compatible with old config files
    nclasses = config['nclasses']
else: 
    nclasses = 751 

if 'ibn' in config:
    ibn = config['ibn']

if 'linear_num' in config:
    linear_num = config['linear_num']

if linear_num <= 0 and (use_swin or use_dense):
    linear_num = 1024

stride = config['stride']

def get_structure():
    if use_swin:
        model_structure = ft_net_swin(nclasses, stride = stride, linear_num=linear_num)
    elif use_dense:
        model_structure = ft_net_dense(nclasses, stride = stride, linear_num=linear_num)
    else:
        model_structure = ft_net(nclasses, stride = stride, ibn = ibn, linear_num=linear_num)
    
    return model_structure
    
def load_network(network):
    
    # netword = model_structure

    save_path = os.path.join('./model',name,'net_%s.pth'%epoch)
    try:
        if use_gpu:
            network.load_state_dict(torch.load(save_path))
        else:
            network.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    except: 
        if use_gpu and torch.cuda.get_device_capability()[0]>6 and len(gpu_ids)==1 and int(version[0])>1: # should be >=7
            print("Compiling model...")
            torch.set_float32_matmul_precision('high')
            network = torch.compile(network, mode="default", dynamic=True) # pytorch 2.0
        if use_gpu:
            network.load_state_dict(torch.load(save_path))
        else:
            network.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
        
# map_location=torch.device('cpu')

    return network

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_feature_from_img(image, model):
    batchsize=32
    # Load and preprocess the image
    # image = Image.open(image_path).convert('RGB')
    # n, c, h, w = image.size()

    image = data_transforms(image).unsqueeze(0)  # Add batch dimension
    # ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

    # Extract features from the image
    model.eval()
    with torch.no_grad():
        features = torch.zeros(1, linear_num).cuda() if use_gpu else torch.zeros(1, linear_num)
        for i in range(2):
            if i == 1:
                # Apply horizontal flipping for augmentation
                image = torch.flip(image, dims=[3])
            input_img = Variable(image.cuda()) if use_gpu else Variable(image)
            for scale in ms:
                if scale != 1:
                    input_img = torch.nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                features += outputs

        # Normalize features
        features /= torch.norm(features, p=2, dim=1, keepdim=True)
        # features = features.cpu()
    return features.cpu()

def compare_images(features1, features2, threshold=0.4):
    # Compute cosine similarity between feature vectors
    # features1_flat = features1.reshape(features1.shape[0], -1)
    # features2_flat = features2.reshape(features2.shape[0], -1)
    similarity_score = 1 - cosine(features1, features2)
    
    # Compare similarity score with threshold
    if similarity_score >= threshold:
        return True  # Images are considered to be of the same person
    else:
        return False  # Images are considered to be of different persons


def extract_feature_from_path(image_path, batchsize=32):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    # n, c, h, w = image.size()

    image = data_transforms(image).unsqueeze(0)  # Add batch dimension
    # ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

    # Extract features from the image
    model.eval()
    with torch.no_grad():
        features = torch.zeros(1, linear_num).cuda() if use_gpu else torch.zeros(1, linear_num)
        for i in range(2):
            if i == 1:
                # Apply horizontal flipping for augmentation
                image = torch.flip(image, dims=[3])
            input_img = Variable(image.cuda()) if use_gpu else Variable(image)
            for scale in ms:
                if scale != 1:
                    input_img = torch.nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                features += outputs

        # Normalize features
        features /= torch.norm(features, p=2, dim=1, keepdim=True)
        # features = features.cpu()
    return features.cpu()

# Test
# print("Test")
# model_structure = get_structure()
# model = load_network(model_structure)
# model.classifier.classifier = nn.Sequential()
# if use_gpu:
#     model = model.cuda()

# with torch.no_grad():
#     features1 = extract_feature_from_path('one.jpg')
#     # gallery_feature = extract_feature(model,dataloaders['gallery'])
#     features2 = extract_feature_from_path('three.jpg')
#     # query_feature = extract_feature(model,dataloaders['query'])
# is_same_person = compare_images(features1, features2, threshold=0.7)

# if is_same_person:
#     print("The images are of the same person.")
# else:
#     print("The images are of different persons.")