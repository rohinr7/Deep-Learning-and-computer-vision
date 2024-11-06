import torch
from PIL import Image
from torchvision import transforms
import json
import cv2
import urllib
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import os
import random

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]



class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))
            # self.handles.append(
            #     target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)
    
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

class Feature_Explanation_method:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer 
        self.ANG = ActivationsAndGradients(model , self.target_layer , None)

    def expand_flat_values_to_activation_shape(self, values, W_layer, H_layer):
        expanded = values.reshape((-1, 1, 1)) * np.ones((1, W_layer, H_layer))  # Ensure shape [N_channels, W_layer, H_layer]
        return expanded

    #first step of the FEM
    def compute_binary_maps(self , activations, K):

        # Calculate mean and std for each channel in the batch
        mean = activations.mean(dim=(2, 3), keepdim=True)  # Mean across height and width
        std = activations.std(dim=(2, 3), keepdim=True)    # Std across height and width
        # print(f"the mean is {mean}")
        # Threshold is mean + K * std for each channel
        threshold = mean + K * std
        # print(f" the threshold is {threshold}")
        # Create binary maps
        binary_maps = (activations >= threshold).float()  # 1.0 for values greater than threshold, 0.0 otherwise

        print(f" binary maps max :{binary_maps.max()} and min is {binary_maps.min()}")

        return binary_maps  
        

    def aggregate_binary_maps(self, binary_feature_map, feature_map):
        # This weigths the binary map based on original feature map
        batch_size, N_channels, W_layer, H_layer = feature_map.shape

        orginal_feature_map = feature_map[0]
        binary_feature_map = binary_feature_map[0]

        channel_weights = orginal_feature_map.mean(dim=(1, 2))
        expanded_weights = self.expand_flat_values_to_activation_shape(channel_weights, W_layer,H_layer)

        print(f" weights of the channel weights: {channel_weights.shape}")
        print(f"the shapeof the expanded weighs : {expanded_weights.shape}")
        
        expanded_feat_map = np.multiply(expanded_weights, binary_feature_map)

        print(f" multiplied maps max :{expanded_feat_map.max()} and min is {expanded_feat_map.min()}") 
        print(f"multiplied : {expanded_feat_map.shape}") 
        
        # Aggregate the feature map of each channel
        feat_map = torch.sum(expanded_feat_map, dim=0)
        print(f"Featmap after sum max :{feat_map.max()} and min is {feat_map.min()}") 

        print(f"feat_map : {feat_map.shape}") 
        feat_map = torch.abs(feat_map)
        heatmap = F.relu(feat_map)
        print(f"Heatmaps after RELU  max :{heatmap.max()} and min is {heatmap.min()}") 


        heatmap /= torch.max(heatmap)
        if len(heatmap.shape) == 3:
            heatmap = heatmap.mean(dim=0)  # Average over the batch or select heatmap[0] for a single one
        heatmap = heatmap.cpu().numpy()
        print(f"shape beofre rescaling: {heatmap.shape}")
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))
        if heatmap_resized.ndim == 2:
            heatmap_resized = np.stack([heatmap_resized] * 3, axis=-1)

        return  heatmap_resized 
    
    def forward(self, input_tensor,targets):
        outputs = self.ANG(input_tensor)
    
        # if targets is None:
        #     target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        #     targets = [ClassifierOutputTarget(category) for category in target_categories]

        # self.model.zero_grad()
        # loss = sum([target(output) for target, output in zip(targets, outputs)])
        # loss.backward(retain_graph=True)

        activations = self.ANG.activations

        #print(activations)
        #gradients = self.ANG.gradients

        for i, activation in enumerate(activations):
            binary_maps = self.compute_binary_maps(activation, K=2)
            saliency = self.aggregate_binary_maps(binary_maps, activation)

        return saliency    

    def __call__(self,input_tensor):
        self.input_tensor = input_tensor
        heatmap = self.forward(self.input_tensor,targets= None)
        return heatmap    
    
# if __name__ == "__main__":
#     # Load a pre-trained ResNet model from torchvision
#     model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#     # print(model)
#     model.eval()

#     inputimages = r"C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\lab4\LIME\data\African_elephant"

#     # paths = os.listdirs(inputimages)
#     paths = random.sample(os.listdir(inputimages), 10)

#     fig, axs = plt.subplots(2, 5, figsize=(15, 6))
#     #fig.suptitle('Grad-CAM Visualizations of Random 10 Images') 
#     fig.suptitle('FEM Visualizations of Random 10 Images') 

#     for idx, path in enumerate(paths):
#         # Load the input image
#         input_image = Image.open(os.path.join(inputimages, path))

#         # Preprocessing the image
#         preprocess = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         input_tensor = preprocess(input_image)
#         input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

#         # Move to GPU if available
#         if torch.cuda.is_available():
#             input_batch = input_batch.to('cuda')
#             model.to('cuda')

#         # Make a prediction
#         with torch.no_grad():
#             output = model(input_batch)

#         # Get probabilities
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)

#         # Download ImageNet labels
#         url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
#         filename = 'imagenet_classes.txt'
#         urllib.request.urlretrieve(url, filename)

#         # Load class names
#         with open(filename, 'r') as f:
#             labels = [line.strip() for line in f.readlines()]

#         # Get the index of the class with the highest probability
#         _, predicted_idx = torch.max(probabilities, dim=0)

#         # Print the predicted class and the probability
#         predicted_class = labels[predicted_idx.item()]
#         predicted_prob = probabilities[predicted_idx].item()

#         print(f"Predicted class: {predicted_class}, Probability: {predicted_prob:.4f}")

#         ########################################################################################################################################
        
#         #########################################################################################################################################

#         #targets = [ClassifierOutputTarget(predicted_idx)]
#         targets = None
#         target_layers = [model.layer4[-1].conv2]

#         fem = Feature_Explanation_method(model, target_layers)
#         heatmap = fem(input_batch, targets)

#         # Load and preprocess the input image (ensure it's of shape (224, 224, 3))
#         input_image = np.array(input_image)
#         input_image = cv2.resize(input_image, (224, 224))
#         input_image = input_image.astype(np.float32) / 255.0  # Normalize the image to range [0, 1]

#         # Overlay the heatmap on the input image
#         visualization = show_cam_on_image(input_image, heatmap, use_rgb=True)

#         ax = axs[idx // 5, idx % 5]
#         ax.imshow(visualization)
#         ax.set_title(f"{predicted_class} ({predicted_prob:.2f})")
#         ax.axis('off')

#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()