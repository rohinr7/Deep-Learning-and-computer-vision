import cv2
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt


def represent_heatmap_overlaid(original_image, heatmap, colormap="jet"):
    """
    Function to overlay a heatmap on an image using a specified colormap.
    """
    # Ensure heatmap is a NumPy array and normalize if necessary
    if not isinstance(heatmap, np.ndarray):
        heatmap = np.array(heatmap)
    # Normalize heatmap to 0-1 if values are not in this range
    if heatmap.max() > 1:
        heatmap = heatmap / heatmap.max()

    # Convert heatmap to a NumPy array, rescale to 0-255, and apply colormap
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to 0-255
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap

    # Convert the original PIL image to a format compatible with OpenCV
    original_image_cv = np.array(original_image)
    if original_image_cv.shape[2] == 3:  # If RGB
        original_image_cv = cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2BGR)

    # Resize the heatmap to match original image dimensions
    heatmap_colored = cv2.resize(heatmap_colored, (original_image_cv.shape[1], original_image_cv.shape[0]))

    # Overlay heatmap on original image
    overlaid_image = cv2.addWeighted(heatmap_colored, 0.8, original_image_cv, 0.5, 0)

    # Convert back to RGB for display with matplotlib
    overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
    return overlaid_image


# def represent_heatmap_overlaid(Saliency, image, cmap):
#     """ 
#     Plots the saliency map with the specified colormap and overlay.
     
#     Parameters:
#     Saliency (numpy.ndarray): 2D array representing the saliency map.
#     image (pil image): image that saliency can overlay
#     Cmap (str): Colormap name to be used for visualization.
    

#     Returns:
#     RGB image

#     """
#     Saliency = (Saliency - np.min(Saliency)) / (np.max(Saliency) - np.min(Saliency))
#     cmap=plt.get_cmap(cmap)

#     heatmap_cl = np.asarray(cmap(Saliency))[:, :, :3]
#     heatmap_cl = Image.fromarray((heatmap_cl * 255).astype(np.uint8))
#     blendimg = Image.blend(image, 
#             heatmap_cl.convert(image.mode), 
#             0.6
#         )    

#     return blendimg
 

def represent_heatmap(Saliency: np.ndarray, Cmap: str) -> np.ndarray:
    """
    Plots the saliency map with the specified colormap and returns the image with the colormap applied.
    
    Parameters:
    Saliency (numpy.ndarray): 2D array representing the saliency map.
    Cmap (str): Colormap name to be used for visualization.
    
    Returns:
    np.ndarray: The RGB image with the colormap applied.
    """
    Saliency = (Saliency - np.min(Saliency)) / (np.max(Saliency) - np.min(Saliency))
    fig, ax = plt.subplots()
    heatmap = ax.imshow(Saliency, cmap=Cmap)
    ax.axis('off')
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    rgb_image = rgb_image.reshape((height, width, 3))
    plt.close(fig)

    return rgb_image




def represent_isolines(saliency: np.ndarray, cmap=None) -> np.ndarray:
    """
    Extract isolines from the saliency map and plot them on a black background.
    
    Parameters:
    saliency (np.ndarray): The saliency map (grayscale image).
    cmap (Union[None, Colormap]): The colormap to apply to the saliency map.
    
    Returns:
    np.ndarray: The RGB image with isolines overlaid on a black background.
    """
    black_bg = np.zeros_like(saliency)
    fig, ax = plt.subplots()
    ax.imshow(black_bg, cmap='gray')
    ax.contour(saliency, cmap=cmap or 'hot', linewidths=1)  
    ax.axis('off')
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    rgb_image = rgb_image.reshape((height, width, 3))
    plt.close(fig)

    return rgb_image




def represent_hard_selection(saliency, image, threshold):
    """ 
    Plots the saliency map with Thresholding.
     
    Parameters:
    Saliency (numpy.ndarray): 2D array representing the saliency map.
    image (pil image): image that saliency can overlay
    Threshold: float number between 0 and 1
    

    Returns:
    RGB image

    """
    image = np.array(image)
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))

    for row in range(saliency.shape[0]):
        for col in range(saliency.shape[1]):
            if saliency[row,col] < threshold:
                image[row,col,:] = 0
            else :
                pass
    
    return image

def represent_soft_selection(saliency, image , threshold):
    """ 
    Plots the saliency map with Thresholding.
     
    Parameters:
    Saliency (numpy.ndarray): 2D array representing the saliency map.
    image (pil image): image that saliency can overlay
    Threshold: float number between 0 and 1
    

    Returns:
    RGB image

    """
    image = np.array(image)
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))

    for row in range(saliency.shape[0]):
        for col in range(saliency.shape[1]):
            image[row,col,:] = image[row,col,:] * saliency[row,col]
    
    
    return image

def visualize_saliency(saliency_map, test_img):
    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Adjust figsize as needed

    # Task 1: Heatmap
    htmap = represent_heatmap(saliency_map, 'hot')
    axs[0, 0].imshow(htmap)
    axs[0, 0].set_title('Saliency Map')
    axs[0, 0].axis('off')  # Hide the axes

    # Task 2: Overlaid Heatmap
    pre_img = represent_heatmap_overlaid(saliency_map, test_img, "hot")
    axs[0, 1].imshow(pre_img)
    axs[0, 1].set_title('Saliency Map overlaid')
    axs[0, 1].axis('off')  # Hide the axes

    # Task 3: Isolines
    iso = represent_isolines(saliency_map, "hot")
    axs[0, 2].imshow(iso)
    axs[0, 2].set_title('Isolines')
    axs[0, 2].axis('off')  # Hide the axes

    # Task 4: Isolines on Test Image
    axs[1, 0].imshow(test_img)
    axs[1, 0].contour(saliency_map, cmap='hot', linewidths=1.5)  # Corrected cmap parameter
    axs[1, 0].set_title('Saliency Map overlaid in Image with Isolines')
    axs[1, 0].axis('off')  # Hide the axes

    # Task 5: Hard Selection (replace with your hard selection function)
    hardimg = represent_hard_selection(saliency_map, test_img, 0.5)
    axs[1, 1].imshow(hardimg)
    axs[1, 1].set_title('Black and Mask')
    axs[1, 1].axis('off')  # Hide the axes

    # Task 6: Soft Selection (replace with your soft selection function)
    softimg = represent_soft_selection(saliency_map, test_img, 0.5)
    axs[1, 2].imshow(softimg)
    axs[1, 2].set_title('Soft Representation')
    axs[1, 2].axis('off')  # Hide the axes

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the combined plot
    plt.show()
     
