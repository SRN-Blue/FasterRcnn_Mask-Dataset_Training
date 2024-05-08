import torch
from bs4 import BeautifulSoup


def generate_box(obj):
    """
    Generate bounding box coordinates from XML object.

    Args:
    - obj (BeautifulSoup): XML object containing bounding box coordinates.

    Returns:
    - list: List containing bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    """
    Generate label for object based on XML object name.

    Args:
    - obj (BeautifulSoup): XML object containing object name.

    Returns:
    - int: Label corresponding to the object name.
    """
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0


def generate_target(image_id, file):
    """
    Generate target annotations for a given image.

    Args:
    - image_id (int): ID of the image.
    - file (str): Path to the XML annotation file.

    Returns:
    - dict: Dictionary containing target annotations (boxes, labels, image_id).
    """
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'lxml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])

        # Annotation dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        return target
