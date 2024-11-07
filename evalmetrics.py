from PIL import Image, ImageFilter
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import load_model, data_transform, device
from rice import get_rice_heatmap
import numpy as np
from sklearn.metrics import auc
from torchvision import transforms
from scipy.stats import pearsonr
import torch.nn.functional as F


def deletion_auc(model, image, input_tensor, saliency, N):
    """
    Computes the Deletion AUC score for a model by progressively removing pixels
    based on the saliency map and calculating the prediction score after each deletion.

    Parameters:
    - model: The model used to make predictions.
    - image: The original PIL image to modify.
    - input_tensor: The input tensor for the model's initial prediction.
    - saliency: An importance map (saliency map) that indicates pixel importance.
    - N: Number of pixels to remove per step.

    Returns:
    - deletion_score: The Area Under the Curve (AUC) for the deletion score.
    """
    if saliency.ndim == 3 and saliency.shape[2] == 3:
        saliency = np.mean(saliency, axis=2)


    # Step 1: Get initial prediction score for the target class (before any deletions)
    output = model(input_tensor)
    #initial_score = output.max(1).values.item()  # Get the maximum score as a scalar
    probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
    initial_score = probabilities.max(1).values.item()
    scores = [initial_score]  # List to store prediction scores after each deletion
    
    # Step 2: Sort pixels by importance (flattened)
    saliency_flat = saliency.flatten()
    sorted_indices = np.argsort(-saliency_flat)  # Sort in descending order of importance
    
    # Step 3: Convert the PIL image to a NumPy array for easier pixel manipulation
    modified_image_np = np.array(image)  # Convert to NumPy array (H, W, C)
    
    # Flatten the saliency map to apply deletions in order
    num_pixels = len(saliency_flat)
    for i in range(0, num_pixels, N):
        # Delete the next N pixels by setting them to black (0)
        pixel_indices = sorted_indices[i:i + N]
        for idx in pixel_indices:
            # Calculate 2D coordinates in the original image shape
            row, col = divmod(idx, modified_image_np.shape[1])
            modified_image_np[row, col] = [0, 0, 0]  # Set pixel to black
        
        # Convert the modified NumPy array back to a PIL Image and then to a tensor
        modified_image_pil = Image.fromarray(modified_image_np)
        modified_input = transforms.ToTensor()(modified_image_pil).unsqueeze(0).to(input_tensor.device)
        
        # Get new prediction score
        output = model(modified_input)
        proba = F.softmax(output, dim=1)  # Convert logits to probabilities
        score = proba.max(1).values.item()
        #score = output.max(1).values.item()  # Get the maximum score as a scalar
        
        # Store the new prediction score
        scores.append(score)
    
    # Step 5: Calculate the AUC (Area Under Curve) based on the scores
    #print(f"the deletion scores are : {scores}")
    deletion_score = auc(np.linspace(0, 1, len(scores)), scores)
    #plot_curve_auc(scores,"deletiion")
    return scores ,deletion_score


def insertion_auc(model, image, input_tensor, saliency, N):
    """
    Computes the Insertion AUC score for a model by progressively adding pixels
    based on the saliency map and calculating the prediction score after each insertion.

    Parameters:
    - model: The model used to make predictions.
    - image: The original PIL image to modify.
    - input_tensor: The input tensor for the model's initial prediction.
    - saliency: An importance map (saliency map) that indicates pixel importance.
    - N: Number of pixels to add per step.

    Returns:
    - insertion_score: The Area Under the Curve (AUC) for the insertion score.
    """
    if saliency.ndim == 3 and saliency.shape[2] == 3:
        saliency = np.mean(saliency, axis=2)

    # Step 1: Create a blurred version of the image as the starting point
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
    modified_image_np = np.array(blurred_image)  # Convert to NumPy array (H, W, C)
    
    # Step 2: Sort pixels by importance (flattened)
    saliency_flat = saliency.flatten()
    sorted_indices = np.argsort(-saliency_flat)  # Sort in descending order of importance
    
    # Step 3: Get initial prediction score for the target class (on the blurred image)
    blurred_input = transforms.ToTensor()(blurred_image).unsqueeze(0).to(input_tensor.device)
    output = model(blurred_input)
    #initial_score = output.max(1).values.item()  # Get the maximum score as a scalar
    probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
    initial_score = probabilities.max(1).values.item()
    scores = [initial_score]  # List to store prediction scores after each insertion
    
    # Flatten the saliency map to apply insertions in order
    num_pixels = len(saliency_flat)
    for i in range(0, num_pixels, N):
        # Add the next N pixels by setting them to the original values
        pixel_indices = sorted_indices[i:i + N]
        for idx in pixel_indices:
            # Calculate 2D coordinates in the original image shape
            row, col = divmod(idx, modified_image_np.shape[1])
            modified_image_np[row, col] = np.array(image)[row, col]  # Set pixel to original
        
        # Convert the modified NumPy array back to a PIL Image and then to a tensor
        modified_image_pil = Image.fromarray(modified_image_np)
        modified_input = transforms.ToTensor()(modified_image_pil).unsqueeze(0).to(input_tensor.device)
        
        # Get new prediction score
        output = model(modified_input)
        probab = F.softmax(output, dim=1)  # Convert logits to probabilities
        score = probab.max(1).values.item()
        # score = output.max(1).values.item()  # Get the maximum score as a scalar
        
        # Store the new prediction score
        scores.append(score)
    
    # Step 5: Calculate the AUC (Area Under Curve) based on the scores
    insertion_score = auc(np.linspace(0, 1, len(scores)), scores)
    #plot_curve_auc(scores ,"insertion")
    return scores, insertion_score


def plot_curve_auc(scores, method):
    """
    Plots the Deletion AUC curve showing how the model's prediction score changes
    as pixels are progressively deleted.
    
    Parameters:
    - scores: List of prediction scores after each deletion step.
    
    Returns:
    - auc_value: The Area Under the Curve (AUC) value for the deletion plot.
    """
    
    # Generate the x-axis as the percentage of deleted pixels
    x = np.linspace(0, 100, len(scores))  # Percentage of deleted pixels (0% to 100%)
    
    # Calculate the AUC using the scores
    auc_value = auc(x / 100, scores)  # Normalize x to be in the range [0, 1]
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, scores, marker='o', color='b', label=f'{method}')
    if method == "insertion":
        plt.xlabel(f'Percentage of Pixels inserted')
    else: 
        plt.xlabel(f'Percentage of Pixels deleted')
    
    plt.ylabel('Prediction Score')
    plt.title(f'{method} Curve (AUC = {auc_value:.4f})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return auc_value


def calculate_pcc(gt, map):
    """Calculates the Pearson correlation coefficient (PCC) between two images.

    Args:
        gt: A numpy array representing the ground truth image.
        map: A numpy array representing the predicted image.

    Returns:
        A float representing the PCC between the two images.
    """
    if map.ndim == 3 and map.shape[2] == 3:
        map = np.mean(map, axis=2)

    gt_flattened = gt.flatten()
    map_flattened = map.flatten()
    pcc, _ = pearsonr(gt_flattened, map_flattened)

    return pcc


def calculate_sim(gt, map):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        gt (np.ndarray): A numpy array representing the ground truth image.
        map (np.ndarray): A numpy array representing the predicted image.

    Returns:
        float: The SSIM between the two images.
    """
    if map.ndim == 3 and map.shape[2] == 3:
        map = np.mean(map, axis=2)

    gt = (gt - gt.min()) / (gt.max() - gt.min())
    gt = gt / np.sum(gt)
    map = (map - map.min()) / (map.max() - map.min())
    map = map / np.sum(map)
    sim = np.sum(np.minimum(gt, map))

    return sim





    
    