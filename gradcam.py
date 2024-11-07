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


