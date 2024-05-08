import torchvision
import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def get_model_instance_segmentation(num_classes):
    """
    Initialize and configure a Faster R-CNN instance segmentation model.

    Args:
    - num_classes (int): Number of classes for detection.

    Returns:
    - model (torchvision.models.detection.FasterRCNN): Configured Faster R-CNN model.
    """
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_model():
    """
    Get the instance segmentation model with custom number of classes.

    Returns:
    - model (torchvision.models.detection.FasterRCNN): Instance segmentation model.
    """
    return get_model_instance_segmentation(3)


def get_trained_model():
    """
    Load and return a trained Faster R-CNN model.

    Returns:
    - model (torch.nn.Module): Trained Faster R-CNN model.
    """
    save_path = r'.\Trained_Model\model4.pt'
    model = torch.load(save_path)
    model.eval()
    return model
