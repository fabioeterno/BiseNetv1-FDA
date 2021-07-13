import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random


def augmentation(image, label):
    #H = random.randint(300,700)
    #W = random.randint(300,900)
    #temp = random.randint(1,5)
    #x = images
    #y = masks
    #if temp == 1:
     #   aug = CenterCrop(H, W, p=1.0)
     #   augmented = aug(image=x, mask=y)
     #   x = augmented["image"]
     #   y = augmented["mask"]
    #if temp == 2:
     #   aug = RandomRotate90(p=1.0)
     #   augmented = aug(image=x, mask=y)
     #   x = augmented['image']
     #   y = augmented['mask']
    #if temp == 3:
     #   aug = GridDistortion(p=1.0)
     #   augmented = aug(image=x, mask=y)
     #   x = augmented['image']
     #   y = augmented['mask']
    #if temp == 4:
     #   aug = HorizontalFlip(p=1.0)
     #   augmented = aug(image=x, mask=y)
     #   x = augmented['image']
     #   y = augmented['mask']
    #if temp == 5:
     #   aug = VerticalFlip(p=1.0)
     #   augmented = aug(image=x, mask=y)
     #   x = augmented['image']
     #   y = augmented['mask']
    ##x = x.resize((3, 720, 960))
    #x = cv2.resize(x, dsize=(960,720), interpolation=cv2.INTER_CUBIC)
    ##y = y.resize((3, 720, 960))
    #y = cv2.resize(y, dsize=(960,720), interpolation=cv2.INTER_CUBIC)

    return image, label


def augmentation_pixel(image):
    ## augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    #temp = random.randint(1,5)
    #if temp == 1:
        #img = image
        #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #h, s, v = cv2.split(hsv)
        #lim = 255 - 30
        #v[v > lim] = 255
        #v[v <= lim] += 30
        #final_hsv = cv2.merge((h, s, v))
        #img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
      
    #if temp == 2:
        #image = Image.fromarray(image)
        #enhancer = ImageEnhance.Contrast(image)
        #factor = (random.randint(1,100)/100)*2
        #img = enhancer.enhance(factor)
        #img = np.array(img)

    #if temp == 3:
        #OriImage  = Image.fromarray(image)
        #img = OriImage.filter(ImageFilter.BLUR)
        #img = np.array(img)

    #if temp == 4:
        #img = Image.fromarray(image)
        ## getting the number of channels
        #channel_count = len(img.getbands())
        #img_arr = np.reshape(img, (img.height, img.width, channel_count))
        ## splitting up channels
        #channels = [img_arr[:,:,x] for x in range(channel_count)]
        ## setting up a shuffling order for rows
        #random_perm = np.random.permutation(img.height) 
        ## reordering the rows with respect to the permutation
        #shuffled_img_arr = np.dstack([x[random_perm, :] for x in channels]).astype(np.uint8)
        #img = shuffled_img_arr
      
    #if temp == 5:
        #im = Image.fromarray(image)
        #im = im.convert('RGB')
        #r, g, b = im.split()
        #r = r.point(lambda i: i * (random.randint(1,100)/100)*2)
        #g = g.point(lambda i: i * (random.randint(1,100)/100)*2)
        #b = b.point(lambda i: i * (random.randint(1,100)/100)*2)
        #img = Image.merge('RGB', (r, g, b))
        #img = np.array(img)
	#image = img
    return image


class CamVid(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, csv_path, scale, loss='dice', mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.png')))
        self.image_list.sort()
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()
        # self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        # self.label_list = [os.path.join(label_path, x + '_L.png') for x in self.image_list]
        self.label_info = get_label_info(csv_path)
        # resize
        # self.resize_label = transforms.Resize(scale, Image.NEAREST)
        # self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        # self.crop = transforms.RandomCrop(scale, pad_if_needed=True)
        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss
        

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])
        # random crop image
        # =====================================
        # w,h = img.size
        # th, tw = self.scale
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)
        # img = F.crop(img, i, j, th, tw)
        # =====================================

        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        # =====================================
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        # =====================================

        img = np.array(img)
        # load label
        label = Image.open(self.label_list[index])


        # crop the corresponding label
        # =====================================
        # label = F.crop(label, i, j, th, tw)
        # =====================================

        # randomly resize label and random crop
        # =====================================
        if self.mode == 'train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        label = np.array(label)


        # augment image and label
        if self.mode == 'train':
            # set a probability of 0.5
            img, label = augmentation(img, label)

        # augment pixel image
        if self.mode == 'train':
            # set a probability of 0.5
            img = augmentation_pixel(img)

        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label).long()

            return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    # data = CamVid('/path/to/CamVid/train', '/path/to/CamVid/train_labels', '/path/to/CamVid/class_dict.csv', (640, 640))
    data = CamVid(['/data/sqy/CamVid/train', '/data/sqy/CamVid/val'],
                  ['/data/sqy/CamVid/train_labels', '/data/sqy/CamVid/val_labels'], '/data/sqy/CamVid/class_dict.csv',
                  (720, 960), loss='crossentropy', mode='val')
    from model.build_BiSeNet import BiSeNet
    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

    label_info = get_label_info('/data/sqy/CamVid/class_dict.csv')
    for i, (img, label) in enumerate(data):
        print(label.size())
        print(torch.max(label))
