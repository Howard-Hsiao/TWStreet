import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
from os.path import basename, splitext
import pandas as pd

img_folder = sys.argv[1]
detection_model_path = sys.argv[2]
temp_folder = "./temp"

class TWStreetIMG(Dataset):
    """
    TWStreet dataset
    """

    def __init__(self, img_folder, name="TWStreet"):
        super(TWStreetIMG, self).__init__()

        self.img_folder = img_folder
        self.filenames = []
        for img_name in os.listdir(self.img_folder):
            img_name = os.path.join(self.img_folder, img_name)
            self.filenames.append(img_name)

        self.img_w = []
        self.img_h = []


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_name = splitext(basename(self.filenames[index]))[0]
        img = Image.open(self.filenames[index]).convert('RGB')
        w, h = img.size
        return (img, img_name, w, h)

dataset = TWStreetIMG(img_folder)
detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=detection_model_path, force_reload=True)
detection_model.eval()


output_folder = temp_folder
img_name_list = []
left_list = []
top_list = []
right_list = []
bottom_list = []
confidence_list = []
class_list = []

for i, (img, img_name, img_w, img_h) in enumerate(dataset):
    results = detection_model(img)
    for j, (left, top, right, bottom, confidence, classes) in enumerate(results.xyxy[0]):
        left, top, right, bottom = left.item(), top.item(), right.item(), bottom.item()
        height = bottom-top
        width  = right-left
        sub_img = transforms.functional.crop(img, top, left, height, width)

        if width < height:
            sub_img = sub_img.rotate(90, expand=True)

        sub_img_name = f"{img_name}-{j}.jpg"

        img_name_list.append(sub_img_name)
        left_list.append(left)
        top_list.append(top)
        right_list.append(right)
        bottom_list.append(bottom)
        confidence_list.append(confidence.item())
        class_list.append(classes.item())

        sub_img.save(os.path.join(output_folder, sub_img_name))
    print(i, img_name)

data = pd.DataFrame({
    "imgName": img_name_list, 
    "x0": left_list, 
    "y0": top_list, 
    "x1": right_list, 
    "y1": top_list, 
    "x2": right_list, 
    "y2": bottom_list, 
    "x3": left_list, 
    "y3": bottom_list, 
    "dec_confidence": confidence_list,
    "class": class_list
})

detection_output_path = os.path.join("output", "YOLOv5_detection.csv")
data.to_csv(detection_output_path, index=False)


print(f"[Detection] The imgs produced by YOLOv5 are store in the folder f{temp_folder}")
print(f"[Detection] The detection result is stored at {detection_output_path}")