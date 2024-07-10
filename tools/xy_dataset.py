import os
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

import cv2
import PIL.Image
import copy
import random

class XYDataset(Dataset):

    def __init__(self, paths, transform=None):
        self.categories = ['throttle', 'steering']
        self.transform = transform

        dataset = []
        for path in paths:
            data_file = os.path.join(path, 'dataset.csv')            
            with open(data_file, 'r') as f:
                for line in f:
                    d = line.replace(' ', '').split(',')
                    dataset.append({
                        'file': os.path.join(path, "images", d[0] + "_front.jpg"),
                        'throttle': float(d[1]), 
                        'steering': float(d[2])
                        })

                    # if int(d[6]) > 0:
                    #     dataset.append({
                    #         'file': os.path.join(path, "images", d[0] + "_front.jpg"),
                    #         'throttle': float(d[1]), 
                    #         'steering': float(d[2])
                    #         })

        self.dataset = dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path = self.dataset[index]['file']
        throttle = self.dataset[index]['throttle']
        steering = self.dataset[index]['steering']
        
        image = cv2.imread(image_path)
        # ######
        # image = cv2.rectangle(image, (0, 0), (640, 130), (0, 0, 0), thickness=-1)
        # ######
        # image = cv2.rectangle(image, (150, 260), (490, 360), (0, 0, 0), thickness=-1)

        if self.transform:
            image = self.transform(image)

        return image, Tensor([throttle, steering])
        # return image, Tensor([steering])
    
    def getData(self, index):
        image_path = self.dataset[index]['file']
        throttle = self.dataset[index]['throttle']
        steering = self.dataset[index]['steering']        
        image = cv2.imread(image_path)

        return image, (throttle, steering)

    def getSample(self, ratio):

        temp1 = copy.deepcopy(self)
        temp2 = copy.deepcopy(self)

        num = len(self.dataset)
        random_id = random.sample(range(num), int(num*ratio))

        temp1_array = []
        temp2_array = []

        for i in range(num):
            if i in random_id:
                temp1_array.append(self.dataset[i])
            else:
                temp2_array.append(self.dataset[i])

        temp1.dataset = temp1_array
        temp2.dataset = temp2_array

        return temp1, temp2
    

import torchvision.transforms as transforms

if __name__ == '__main__':

    TRANSFORMS = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = XYDataset(['data/1030_09/'], TRANSFORMS)

    train_dataset, eval_dataset = dataset.getSample(0.8)

    print(len(dataset), len(train_dataset), len(eval_dataset))