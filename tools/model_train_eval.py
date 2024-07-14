import argparse

import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader

# from torch2trt import torch2trt
from xy_dataset import XYDataset

import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model train and eval')
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', required=True, type=str)
    parser.add_argument('--mask', '-m', type=str)
    parser.add_argument('--data', '-d', action='append', required=True, type=str)
    parser.add_argument('--epoch', '-e', default=100, type=int)
    parser.add_argument('--batch', '-b', default=8, type=int)
    parser.add_argument('--ratio', '-r', default=0.8, type=float)
    args = parser.parse_args()

    device = torch.device('cuda')
    model = None
    if not args.input:
        model = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(512, 2)
        # model.fc = torch.nn.Linear(512, 1)
    else:
        model = torch.load(args.input)
    model = model.to(device)

    TRANSFORMS = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # トレーニング
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters())

    # 初期の値を取得
    dataset = XYDataset(args.data, TRANSFORMS, args.mask)
    epoch = args.epoch    
    batch_size = args.batch
    ratio = args.ratio

    train_dataset, eval_dataset = dataset.getSample(ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    

    print(f'{len(dataset)}: {len(train_dataset)}/{len(eval_dataset)}')

    try:
        while epoch > 0:
            count = 0
            sum_loss = 0.0
            for images, xy, _, _ in iter(train_loader):
                images = images.to(device)
                xy = xy.to(device)

                optimizer.zero_grad()

                outputs = model(images)            

                loss = 0.0
                for i in range(len(images)):
                    loss += torch.mean((outputs[i] - xy[i])**2)
                loss.backward()
                optimizer.step()

                sum_loss += float(loss)
                count += len(images)
                
                print(f'[{epoch}] {count}/{len(dataset)}: sum_loss={sum_loss/count}, loss={float(loss)}')
            
            epoch -= 1
    except KeyboardInterrupt:
        pass

    # 評価
    model = model.eval()

    winname = 'data_view_1'
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('ID', winname, 0, len(eval_dataset)-1, lambda val: {})

    id = -1
    while True:
        current_id = cv2.getTrackbarPos('ID', winname)
        if current_id != id:
            id = current_id
            res_image, res_xy, _, _ = eval_dataset.getData(id)

            image, xy, _, _ = dataset[id]
            image = image.unsqueeze(dim=0)
            image = image.to(device)       
            output = model(image)
            output = output.to('cpu')

        v_val = float(res_xy[0])
        u_val = float(res_xy[1])        

        v = int((1.0 - v_val) * res_image.shape[0] / 2.0)
        # v = int(res_image.shape[0] / 2.0)
        u = int((1.0 - u_val) * res_image.shape[1] / 2.0)
        
        res_image = cv2.circle(res_image, (u, v), 10, (255, 0, 0), thickness=-1)

        out_v_val = output[0][0]
        out_u_val = output[0][1]
        # out_u_val = output[0]
        out_v = int((1.0 - out_v_val) * res_image.shape[0] / 2.0)
        # out_v = int(res_image.shape[0] / 2.0)
        out_u = int((1.0 - out_u_val) * res_image.shape[1] / 2.0)

        res_image = cv2.circle(res_image, (out_u, out_v), 10, (0, 0, 255), thickness=-1)        
        cv2.imshow(winname, res_image)
        key = cv2.waitKey(30)        
        if key == 115:
            print(f'model save: {args.output}')
            torch.save(model, args.output)
            break
        elif key == 113:
            break

# data = torch.zeros((1, 3, 224, 224)).cuda().half()
# model_trt = torch2trt(model, [data], fp16_mode=True)
# model.load_state_dict(torch.load('road_following_model.pth'))