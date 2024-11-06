import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import numpy as np
from skimage.transform import resize
from tqdm import tqdm  # Use `from tqdm import tqdm` to import the `tqdm` function directly
# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define the transformation to be applied to input images
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def generate_masks(N, s, p1, input_size):
    """
    Generates N random binary masks to perturb the input image for RISE saliency.
    reference : https://github.com/eclique/RISE/blob/master/explanations.py

    Args:
        N (int): Number of masks to generate.
        s (int): Size of the grid in the mask.
        p1 (float): Probability of masking each grid cell.
        input_size (tuple): Size of the input image (height, width).
    
    Returns:
        np.array: Array of generated binary masks.
    """
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size
    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')
    masks = np.empty((N, *input_size))

    for i in tqdm(range(N), desc='Generating filters'):
        # Random shifts for each mask
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        
        # Linear upsampling and cropping the mask
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
    return masks


# Function to apply the masks to the input image
def apply_masks(input_image, masks):
    """
    Applies the generated masks to the input image.
    
    Args:
        input_image (np.array): Original input image (single image in the batch).
        masks (np.array): Array of generated masks.
    
    Returns:
        np.array: Array of perturbed images created by applying masks to the input image.
    """
    # Extend the mask for all 3 channels (RGB) for element-wise multiplication
    perturbed_images = [input_image * mask[..., np.newaxis] for mask in masks]
    return np.array(perturbed_images)

# Function to compute RISE saliency map
def compute_rise_saliency(model, img_tensor, masks, target_class_index, batch_size=100):
    """
    Computes the RISE saliency map by applying random masks and evaluating the effect 
    on the model's output for the target class.
    
    Args:
        model: Pre-trained model for predictions.
        img_tensor (torch.Tensor): Preprocessed image tensor (1, 3, 224, 224).
        masks (torch.Tensor): Tensor of random binary masks with shape [N, 1, H, W].
        target_class_index (int): Index of the target class (class to explain).
        batch_size (int): Number of perturbed images to process at once to manage memory.
    
    Returns:
        torch.Tensor: Computed saliency map of the same size as the input image.
    """
    saliency_map = torch.zeros(img_tensor.size(2), img_tensor.size(3), device=img_tensor.device)

    # Process in batches
    for i in range(0, masks.size(0), batch_size):
        mask_batch = masks[i:i + batch_size]  # Get the current batch of masks
        img_tensor_batch = img_tensor.repeat(mask_batch.size(0), 1, 1, 1)  # Shape: [N, 3, 224, 224]
        perturbed_images = img_tensor_batch * mask_batch  # Broadcasting masks over the batch

        with torch.no_grad():
            perturbed_preds = model(perturbed_images)

        target_scores = perturbed_preds[:, target_class_index]

        # Accumulate contributions to the saliency map
        for j in range(mask_batch.size(0)):
            saliency_map += mask_batch[j][0] * target_scores[j]

    # Normalize the saliency map
    saliency_map /= len(masks)

    return saliency_map


def predict_image(input_tensor, model):
    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
        
    return predicted, confidence



def get_rice_heatmap(model, input_tensor):
    input_array = input_tensor.cpu().numpy()
    input_array = input_array.reshape((224, 224, 3)) 
    print(input_array.shape)
    img_size = (224, 224)
    
    # Generate random binary masks
    N = 1000  # Number of masks
    s = 8  # Mask size (in terms of grid cells)
    p1 = 0.5  # Probability of masking each grid cell
    pred_index, confidence = predict_image(input_tensor, model)
    
    # Generate masks as a PyTorch tensor
    masks = generate_masks(N=N, s=s, p1=p1, input_size=img_size)
    masks_tensor = torch.tensor(masks, dtype=torch.float32, device=input_tensor.device)
    
    # Reshape masks to [N, 1, 224, 224] for compatibility
    masks_tensor = masks_tensor.unsqueeze(1)  # Add a channel dimension

    # Compute the RISE saliency map for the predicted class
    saliency_map = compute_rise_saliency(model, input_tensor, masks_tensor, pred_index)
    # Convert saliency map from PyTorch tensor to NumPy array
    saliency_map_numpy = saliency_map.cpu().numpy()
    
    print(f"Shape of the saliency RICE: {saliency_map_numpy.shape}")

    # Normalize the saliency map to the range [0, 1]
    saliency_map_numpy_min = saliency_map_numpy.min()
    saliency_map_numpy_max = saliency_map_numpy.max()
    
    # Avoid division by zero
    if saliency_map_numpy_max > saliency_map_numpy_min:
        saliency_map_numpy = (saliency_map_numpy - saliency_map_numpy_min) / (saliency_map_numpy_max - saliency_map_numpy_min)
    else:
        saliency_map_numpy = np.zeros_like(saliency_map_numpy)  # If all values are the same, set to zero

    return saliency_map_numpy