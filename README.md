# Face Mask Detection using Faster R-CNN

## Introduction

The integration of computer vision and deep learning has led to significant advancements in object detection tasks, including face mask detection. This project presents an implementation of a Faster R-CNN (Region-based Convolutional Neural Network) model using PyTorch for precise face mask detection and classification. Utilizing a custom dataset sourced from Kaggle, the model accurately identifies faces with masks, masks worn incorrectly, and faces without masks. Augmentation techniques via Albumentations and tailored preprocessing pipelines enhance model robustness, ensuring reliable performance across diverse scenarios. This README provides an overview of the dataset, model architecture, training procedures, evaluation metrics, and usage guidelines, showcasing the project's technical capabilities and contributions to object detection tasks. Let’s explore the detailed technical aspects and outcomes of this impactful project.


## Technologies Used

1. **PyTorch**: A deep learning framework that provides a flexible and efficient platform for building and training neural network models.
2. **Faster R-CNN**: A popular two-stage object detection model that combines region proposal networks with a classifier to detect objects with bounding boxes.
3. **Albumentations**: An image augmentation library that offers a wide range of transformations to augment and diversify the training dataset, improving model generalization.

## Project Workflow

1. **Preprocessing Images**:
   - Resize and normalize images to a consistent size suitable for model input.
   - Apply augmentation techniques such as rotation, flipping, and brightness adjustments using Albumentations to increase dataset diversity.

2. **Data Format Creation**:
   - Generate annotations with bounding box coordinates and class labels for each image in a format compatible with Faster R-CNN training requirements (e.g., COCO format).

3. **Training the Model**:
   - Implement the Faster R-CNN architecture using PyTorch, including backbone networks (e.g., ResNet) for feature extraction and region proposal networks (RPN) for object detection.
   - Train the model using the annotated dataset, optimizing parameters to minimize detection and classification loss functions.

4. **Evaluation and Results**:
   - Evaluate the trained model using a separate validation/test dataset to calculate performance metrics such as Mean Average Precision (mAP), Precision, Recall, and F1-score for each class.
   - Plot and visualize the evaluation results using tools like Matplotlib to analyze model performance and identify areas for improvement.

# Pre Process Steps

## 1. df_creator.py

### Purpose:
This module (`df_creator.py`) creates a DataFrame containing file names from a specified directory containing image files.

### Functionality:
- Utilizes the `os` and `pandas` libraries.
- Uses `os.listdir()` to list all files in the specified image directory.
- Extracts file names without extensions using list comprehensions.
- Creates a DataFrame from the list of file names.

## 2. Extract_data_from_xml.py

### Purpose:
This module (`Extract_data_from_xml.py`) extracts data from XML annotation files related to object bounding boxes and labels.

### Functionality:
- Utilizes the `torch` library for tensor operations and `BeautifulSoup` for XML parsing.
- Implements functions to generate bounding box coordinates and labels from XML objects.
- Generates target annotations (boxes, labels, image_id) for a given image based on its XML annotation file.
- Converts bounding boxes and labels to PyTorch tensors for compatibility with training.

These modules are crucial for preprocessing data and extracting target annotations required for training the Faster R-CNN model for face mask detection. Incorporate these functionalities into your project for structured data handling and annotation extraction during model training and evaluation.

# Dataset

## 1. data_loader.py

### Purpose:
This module (`data_loader.py`) contains functions and configurations for creating PyTorch data loaders for training and evaluation.

### Functionality:
- Defines data transformations using `transforms.Compose()` for preprocessing and augmentation.
- Implements custom augmentation transformations using Albumentations library (`A.Compose()`).
- Provides functions to create training and evaluation data loaders (`get_trainer_dataloader()` and `get_eval_dataloader()`).
- Defines a custom collate function (`collate_fn()`) to handle batched data.

## 2. pytorch_dataset.py

### Purpose:
This module (`pytorch_dataset.py`) defines the `MaskDataset` class for handling the custom dataset for face mask detection.

### Functionality:
- Loads image files and corresponding XML annotation files using provided paths.
- Implements data preprocessing and augmentation using Albumentations library.
- Generates target annotations (bounding boxes and labels) using the `generate_target()` function from `Extract_data_from_xml.py`.
- Implements `__getitem__()` and `__len__()` methods required for dataset handling in PyTorch.
- Provides a `plotter_pre()` method to visualize preprocessed images with bounding box annotations.

### Usage:
1. **`data_loader.py`**:
   - Define data transformations using `data_transform`, `get_train_transform()`, and `get_valid_transform()` functions.
   - Create training and evaluation data loaders using `get_trainer_dataloader()` and `get_eval_dataloader()` functions.

2. **`pytorch_dataset.py`**:
   - Initialize `MaskDataset` class with the DataFrame containing file names.
   - Use `plotter_pre()` method to visualize preprocessed images with bounding box annotations for verification.

These modules facilitate data loading, preprocessing, augmentation, and target annotation generation essential for training and evaluating the Faster R-CNN model for face mask detection.

# Model

## Model

### Purpose:
The `model.py` file contains functions to initialize, configure, and load a Faster R-CNN instance segmentation model for face mask detection.

### Functionality:

#### `get_model_instance_segmentation(num_classes)`
- Initializes and configures a Faster R-CNN instance segmentation model.
- Loads an instance segmentation model pre-trained on COCO dataset.
- Replaces the pre-trained head with a new one for custom classification with the specified number of classes.

#### `get_model()`
- Retrieves the instance segmentation model with a custom number of classes (in this case, 3 classes for face mask detection).

#### `get_trained_model()`
- Loads and returns a trained Faster R-CNN model.
- Assumes a trained model is saved at a specified path (`.\Trained_Model\model4.pt` in this case).
- Sets the model to evaluation mode (`model.eval()`) for inference.

### Usage:
1. **Model Initialization**:
   - Use `get_model_instance_segmentation(num_classes)` to initialize a Faster R-CNN model with a custom number of classes.
   - Alternatively, use `get_model()` to get the instance segmentation model with predefined 3 classes for face mask detection.

2. **Loading Trained Model**:
   - Use `get_trained_model()` to load a pre-trained Faster R-CNN model for inference or further training.
   - Ensure that the trained model is saved at the specified path (`.\Trained_Model\model4.pt`).

### Example Code Snippet:
```python
from Model.model import get_model_instance_segmentation, get_trained_model

# Initialize model for training with 3 classes
model = get_model_instance_segmentation(num_classes=3)

# Load a trained model for inference
trained_model = get_trained_model()
```
  
# tran.py

## Trainer

### Purpose:
The `trainer.py` module is designed to train the Faster R-CNN model for face mask detection using the provided dataset and model configurations.

### Functionality:

- **SummaryWriter Initialization**:
  - Creates a `SummaryWriter` instance to log training information and visualize it using TensorBoard.

- **Data Loading and Model Initialization**:
  - Loads training data using `get_trainer_dataloader()` from the `data_loader.py` module.
  - Initializes the Faster R-CNN model using `get_model()` from the `model.py` module.

- **Device Configuration**:
  - Checks for GPU availability and moves the model to the appropriate device (CPU or GPU).

- **Training Loop**:
  - Runs the training loop for the specified number of epochs (`num_epochs`).
  - Utilizes the Adam optimizer with a specified learning rate and weight decay.
  - Implements a learning rate scheduler (`StepLR`) for adjusting the learning rate during training.

- **Logging and Monitoring**:
  - Logs training loss, epoch details, and iteration details during training.
  - Utilizes the `SummaryWriter` to save logs in the 'runs' directory for visualization in TensorBoard.

- **Model Saving**:
  - Saves the trained model to the specified path (`.\Trained_Model\model.pt`) after training completion.

### Usage:
1. **Training Configuration**:
   - Adjust hyperparameters such as learning rate (`lr`), weight decay (`weight_decay`), and number of epochs (`num_epochs`) as needed.
   
2. **Training Execution**:
   - Run `trainer.py` to initiate the training process for the Faster R-CNN model.
   - Monitor training progress using printed logs and visualize metrics using TensorBoard.

### Example Code Snippet:
```bash
python trainer.py
```

