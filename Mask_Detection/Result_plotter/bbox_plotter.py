import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_image(img_tensor, annotation):
    """
    Plot an image with bounding box annotations.

    Args:
    - img_tensor (torch.Tensor): Input image tensor.
    - annotation (dict): Dictionary containing 'boxes' and 'labels' keys for annotations.

    Returns:
    - None
    """
    fig, ax = plt.subplots(1)

    # Move the image tensor from GPU to CPU and convert it to a numpy array
    img = img_tensor.cpu().detach().numpy()

    # Transpose the image array to change dimensions from (C, H, W) to (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # Display the image
    ax.imshow(img)

    # Move annotation boxes to CPU and convert them to numpy arrays
    boxes = annotation["boxes"].cpu().detach().numpy()
    labels = annotation["labels"].cpu().detach().numpy()

    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Annotate the box with the label
        label = f'Class: {labels[i]}'
        ax.text(xmin, ymin, label, color='r', verticalalignment='top',
                bbox={'color': 'r', 'alpha': 0.5, 'pad': 2})

        # Uncomment the following break statement to plot only the first box
        # break

    plt.show()
