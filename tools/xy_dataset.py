import os
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

import cv2
import PIL.Image
import copy
import random

class XYDataset(Dataset):

    def __init__(self, paths, transform=None, mask_path=None):
        self.categories = ['throttle', 'steering']
        self.transform = transform

        dataset = []
        for path in paths:
            data_file = os.path.join(path, 'dataset.csv')            
            with open(data_file, 'r') as f:
                for line in f:
                    d = line.replace(' ', '').split(',')

                    use = True
                    if len(d) == 4:
                        if int(d[3]) > 0:
                            use = True
                        else:
                            use = False

                    p = d[0].split("/")
                    filepath = ""                    
                    if len(p) == 1:
                        filepath = os.path.join(path, "images", p[0] + "_front.jpg")
                    elif len(p) == 2:
                        filepath = os.path.join(path, "images", p[0], p[1] + "_front.jpg")

                    if use:
                        dataset.append({                        
                            'file': filepath,
                            'name': d[0],
                            'throttle': float(d[1]), 
                            'steering': float(d[2])
                            })

        self.dataset = dataset
        self.mask_path = mask_path


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):        
        image_path = self.dataset[index]['file']
        name = self.dataset[index]['name']
        throttle = self.dataset[index]['throttle']
        steering = self.dataset[index]['steering']
        
        image = cv2.imread(image_path)

        if self.mask_path:
            mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
            image[:, :] = image[:, :] * (1 - mask[:,:,3:] / 255) \
                        + mask[:,:,:3] * mask[:,:,3:] / 255            

        if self.transform:
            image = self.transform(image)

        return image, Tensor([throttle, steering]), name, image_path
    
    def getData(self, index):
        image_path = self.dataset[index]['file']
        name = self.dataset[index]['name']
        throttle = self.dataset[index]['throttle']
        steering = self.dataset[index]['steering']        
        image = cv2.imread(image_path)

        return image, (throttle, steering), name, image_path

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