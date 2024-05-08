import os
import torch

from Dataset.data_loader import get_trainer_dataloader, get_eval_dataloader
from Model.model import get_trained_model
from Result_plotter.bbox_plotter import plot_image


# Get data loaders and trained model
data_loader = get_eval_dataloader()
model = get_trained_model()

# Set environment variable to avoid duplicate library loading error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Determine device (CPU or GPU)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load a batch of images and annotations
for imgs, annotations in data_loader:
    imgs = [img.to(device) for img in imgs]  # Move images to device
    annotations = [{k: v.to(device) for k, v in t.items()}
                   for t in annotations]  # Move annotations to device
    break  # Process only one batch

# Set model to evaluation mode
model.eval()

# Perform inference on the images
preds = model(imgs)

# Display predictions and ground truth for one image
print("Prediction")
plot_image(imgs[2], preds[2])  # Display prediction for the third image
print("Target")
plot_image(imgs[2], annotations[2])  # Display ground truth for the third image
