from albumentations.pytorch import ToTensorV2
from albumentations import Compose, RandomCrop, Normalize
from torchvision import transforms
from PIL import Image
import albumentations as A
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from Pre_Process.Extract_data_from_xml import generate_target

# Define the MaskDataset class for handling the custom dataset


class MaskDataset(object):
    def __init__(self, df, transforms=None):
        self.transforms = transforms
        # Load all image files, sorting them to ensure alignment
        self.imgs = list(
            sorted(os.listdir(r".\Mask_Detection\archive_3\images")))
        self.img_dir = r'.\Mask_Detection\archive_3\images'
        self.annot_dir = r'.\Mask_Detection\archive_3\annotations'
        self.df = df

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename'] + '.png'
        img_path = os.path.join(self.img_dir, img_name)
        annot_path = os.path.join(
            self.annot_dir, img_name.replace('.png', '.xml'))
        # Load image
        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0

        # Generate Label (replace generate_target with your actual function)
        target = generate_target(idx, annot_path)

        if self.transforms:
            # Apply transformations
            sample = {'image': img,
                      'bboxes': target['boxes'], 'labels': target['labels']}
            sample = self.transforms(**sample)
            img = sample['image']
            target['boxes'] = torch.tensor(
                sample['bboxes'], dtype=torch.float32)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def plotter_pre(self, img, target):
        # Create a figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(img)

        # Loop through bounding boxes and add them as patches to the plot
        for bbox, label in zip(target['boxes'], target['labels']):
            xmin, ymin, xmax, ymax = bbox
            rect = patches.Rectangle(
                (xmin * img.shape[1], ymin * img.shape[0]),
                (xmax - xmin) * img.shape[1], (ymax - ymin) * img.shape[0],
                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, label, color='r', verticalalignment='top',
                    bbox={'color': 'r', 'alpha': 0.5, 'pad': 2})

        # Show the plot
        plt.show()
