import numpy
import torch
import glob
import os
from torchvision import transforms
import torchvision.transforms as TF
import torchvision.transforms.functional as func
from torchvision.utils import save_image
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append(os.pardir)
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random



palette = [  
            0,   0,   0,    # Background = 0
          119,  11,  32,    # Bycicle = 1
           70,  70,  70,    # Building = 2
            0,   0, 142,    # Car = 3
          153, 153, 153,    # Pole = 4 
          190, 153, 153,    # Fence = 5
          220,  20,  60,    # Pedestrian = 6
          128,  64, 128,    # RoadLine = 7
          244,  35, 232,    # Sidewalk = 8
          220, 220,   0,    # Traffic sign = 9
           70, 130, 180,    # Sky = 10
          107, 142,  35     # Vegetation = 11      
           ]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

#print("printing palette")      # Print the numpy array of 768 values for checking
#print(palette)

def boolstr_to_floatstr(v):
    if v == 'True':
        return '1'
    elif v == 'False':
        return '0'
    else:
        return v


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    #print("label array:")      # Debugging mask shape
    #print(np.array(new_mask))
    #print("display mask")
    display(new_mask)           # Printing the mask colorized with putpalette
    return new_mask



class IDDA(torch.utils.data.Dataset):
    def __init__(self, image_path = ['/content/drive/MyDrive/Politecnico/Machine Learning/BiseNetv1/dataset/IDDA/rgb'], label_path = ['/content/drive/MyDrive/Politecnico/Machine Learning/BiseNetv1/dataset/IDDA/labels'], csv_path = '/content/drive/MyDrive/Politecnico/Machine Learning/BiseNetv1/dataset/CamVid/class_dict.csv', scale = (720,960), num_classes = 12):
        super().__init__()
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            # list of the training images
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.jpg')))
        self.image_list.sort()
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            # list of the labels
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()
        self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        self.image_size = scale
        self.num_classes = num_classes
        

        # Mapping labels from IDDA dataset to CamVid with a dictionary
        # The keys are the IDDA labels, the values are the CamVid labels that I want to retrieve from IDDA
        self.id_to_trainid = {1:1, 2:4, 4:5, 5:3, 6:6, 7:6, 8:7, 9:10, 10:2, 12:8, 16:0, 20:9, 24:1}
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __getitem__(self, index):
        # Random crop/resizing to implement like in CamVid
        seed = random.random()

        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # Loading RGB image
        img = Image.open(self.image_list[index])
        img = transforms.Resize(scale, Image.BILINEAR)(img)
        img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        img = np.array(img)
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()
        #for i in range(3):
        #  img[i,:,:] = img[i,:,:] - self.mean[i]     #for normalizing
        #  img[i,:,:] = img[i,:,:]/self.std[i]
        #reshaping

        #img = img.permute(1,2,0)
        
        #file_name = self.image_list[index].split("/")[-1]
 
        
        # Loading Ground Truth labels
        label = Image.open(self.label_list[index]).convert('RGB')
        label = transforms.Resize(scale, Image.NEAREST)(label)
        label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        label = np.array(label)

        # I start defining a numpy array with the same shape of the label array (3, 960, 720)
        # The default value here is 0, the Background.
        label_copy = 11*np.ones(label.shape, dtype=np.float32)
        # I iterate over the dictionary, for all the pixels which have the same value of k (I use the boolean mask label == k),
        # I substitute the value of the dictionary, so the CamVid label.
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        # The 3 channels are equal, I need only 1 which contains the value from 0 to 11
        labelToPass = label_copy[:,:,0]
        #print("after mapping")
        #print(labelToPass)
        #label = TF.ToTensor()(label).float()
        label_classes = np.zeros((12,self.image_size[0],self.image_size[1]), dtype=np.float32)
        classes_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #for i in range(0,self.num_classes):
        for i in classes_id:
          #print(i)
          temp = (labelToPass==i)
          #print(temp)
          label_classes[i,:,:] = np.vectorize(boolstr_to_floatstr)(temp).astype(float)
          #print(label_classes[i,:,:])
        labelToPass = label_classes

        #print(labelToPass)
        

        labelToPass = torch.from_numpy(labelToPass)




        # I return the image as Tensor of torch.Size([3, 1080, 1920]), 
        # the labelToPass which is a numpy array of (1080, 1920) where we have the new labels based on Camvid 
        # the label which is a numpy array of (1080, 1920) where we have the labels of IDDA, passing just for checking purpose, not needed of course
        # to delete in the final version
        return img, labelToPass

    def __len__(self):
        return len(self.image_list)



if __name__ == '__main__':
  
    #print("*********** inizio main **************")
    data = IDDA()


    #print("*********** fine main **************")
    fig = plt.figure(figsize=(40, 30))
    for i, (img, label) in enumerate(data):
        #print("printing image")
        #print(img.shape)       # Debugging image shape
        #print(img)

        #print(label.shape)
        #print(label)
        #print(torch.max(label))



        # Colorize the mask 
       # colors_mask = colorize_mask(label)
        #print(colors_mask)


        #img = torchvision.utils.make_grid(img).numpy()
        #img = np.transpose(img, (1, 2, 0))
        #img = img[:, :, ::-1]
        #plt.imshow(img)
        

        # incremente i to print more images
        if(i == 0):
          break