from torchvision import transforms
import torch
import albumentations as A

# Importing custom functions and classes
from Pre_Process.df_creator import get_dataframe
from Dataset.pytorch_dataset import MaskDataset
from albumentations.pytorch import ToTensorV2

# Define the transformations for data preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
])

# Define augmentation transformations for training data


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Random horizontal flip with probability 0.5
        # Random 90-degree rotation with probability 0.5
        A.RandomRotate90(p=0.5),
        # Random brightness and contrast adjustment
        A.RandomBrightnessContrast(p=0.2),
        A.RandomScale(p=0.5),  # Random scaling with probability 0.5
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0, hue=0),  # Color jittering
        A.GaussianBlur(blur_limit=(3, 7)),  # Apply mild Gaussian blur
        ToTensorV2(p=1.0)  # Convert augmented image to PyTorch tensor
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# Define transformations for validation data (no augmentation)


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)  # Convert image to PyTorch tensor
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# Custom collate function to handle batched data


def collate_fn(batch):
    return tuple(zip(*batch))


# Get dataframe from custom function
df = get_dataframe()

# Define training data loader


def get_trainer_dataloader():
    batch_size = 64
    # Create dataset with training transformations
    dataset = MaskDataset(df, get_train_transform())
    data_loader = torch.utils.data.DataLoader(
        # Create data loader with collate function
        dataset, batch_size=batch_size, collate_fn=collate_fn)
    return data_loader

# Define evaluation data loader


def get_eval_dataloader():
    batch_size = 4
    # Create dataset with validation transformations
    dataset = MaskDataset(df, get_valid_transform())
    data_loader = torch.utils.data.DataLoader(
        # Create data loader with collate function
        dataset, batch_size=batch_size, collate_fn=collate_fn)
    return data_loader
