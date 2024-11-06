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
    

class Gradcam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer 
        self.ANG = ActivationsAndGradients(model , target_layer,None)
    
    def forward(self, input_tensor,targets):
        input_ten = torch.autograd.Variable(input_tensor, requires_grad=True)
        outputs = self.ANG(input_ten)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        self.model.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, outputs)])
        loss.backward(retain_graph=True)

        activations = self.ANG.activations
        gradients = self.ANG.gradients

        
        if isinstance(activations, list):
            activations = torch.stack(activations)
        pooled_gradients = torch.stack([grad.flatten(2).mean(-1) for grad in gradients])  # Stack into a tensor
        for i in range(activations.size()[1]):
           activations[:, i, :, :] *= pooled_gradients[:, i].unsqueeze(-1).unsqueeze(-1)
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        if len(heatmap.shape) == 3:
            heatmap = heatmap.mean(dim=0)  # Average over the batch or select heatmap[0] for a single one
        heatmap = heatmap.cpu().numpy()
        self.heatmap_resized = cv2.resize(heatmap, (224, 224))
        self.heatmap_resized = (self.heatmap_resized - np.min(self.heatmap_resized)) / (np.max(self.heatmap_resized) - np.min(self.heatmap_resized))
        if self.heatmap_resized.ndim == 2:
            self.heatmap_resized = np.stack([self.heatmap_resized] * 3, axis=-1)

        return  self.heatmap_resized   
        

    def __call__(self,input_tensor,targets):
        self.input_tensor = input_tensor
        heatmap = self.forward(self.input_tensor,targets)
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
#     fig.suptitle('Grad-CAM Visualizations of Random 10 Images') 
#     #fig.suptitle('FEM Visualizations of Random 10 Images') 

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

#         fem = Gradcam(model, target_layers)
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


# obj = ActivationsAndGradients(model , target_layers,None)

# input_ten = torch.autograd.Variable(input_batch, requires_grad=True)

# outputs = obj(input_ten)

# if targets is None:
#     target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
#     targets = [ClassifierOutputTarget(category) for category in target_categories]


# model.zero_grad()
# loss = sum([target(output) for target, output in zip(targets, outputs)])
# loss.backward(retain_graph=True)

# activations = obj.activations
# gradients = obj.gradients

# # If activations are a list of tensors, stack them into a single tensor
# if isinstance(activations, list):
#     activations = torch.stack(activations)

# # Pool the gradients across the channels
# pooled_gradients = torch.stack([grad.flatten(2).mean(-1) for grad in gradients])  # Stack into a tensor

# # Weight the channels by corresponding gradients
# for i in range(activations.size()[1]):
#     activations[:, i, :, :] *= pooled_gradients[:, i].unsqueeze(-1).unsqueeze(-1)

# # Average the channels of the activations to get the heatmap
# heatmap = torch.mean(activations, dim=1).squeeze()

# # Apply ReLU on top of the heatmap
# heatmap = F.relu(heatmap)

# # Normalize the heatmap
# heatmap /= torch.max(heatmap)

# # If the heatmap is still 3D (batch size present), average over the batch as well or select one
# if len(heatmap.shape) == 3:
#     heatmap = heatmap.mean(dim=0)  # Average over the batch or select heatmap[0] for a single one

# # Convert heatmap to numpy
# heatmap = heatmap.cpu().numpy()

# # Resize the heatmap to match the input image size (224x224)
# heatmap_resized = cv2.resize(heatmap, (224, 224))

# # Normalize the heatmap between 0 and 1 for better visualization
# heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))

# # Convert to 3D if it's 2D, by stacking it across 3 channels
# if heatmap_resized.ndim == 2:
#     heatmap_resized = np.stack([heatmap_resized] * 3, axis=-1)















# gradient = obj.gradients
# print(gradient)

# # for i in gradient:
# #     print(F"shape of the gradient is :  {i.shape}")



##################################################################################################################################################


# # Convert the input PIL image to a NumPy array
# input_image = np.array(input_image)
# input_image = cv2.resize(input_image, (224,224))
# # Normalize the image to range [0, 1] and ensure it's of type np.float32
# input_image = input_image.astype(np.float32) / 255.0

# target_layers = [model.layer4]
# #input_tensor = 
# targets = [ClassifierOutputTarget(predicted_idx)]

# # Construct the CAM object once, and then re-use it on many images.
# # with GradCAM(model=model, target_layers=target_layers) as cam:

# cam = GradCAM(model=model, target_layers=target_layers)

# # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# grayscale_cam = cam(input_tensor=input_batch, targets=targets)
# # In this example grayscale_cam has only one image in the batch:
# grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
# # You can also get the model outputs without having to redo inference
# model_outputs = cam.outputs

# plt.imshow(visualization)
# plt.axis('off')  # Turn off axis labels for cleaner visualization
# plt.show()