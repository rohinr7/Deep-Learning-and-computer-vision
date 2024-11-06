import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import os
from lime import lime_image
from torchvision import transforms
from skimage.segmentation import mark_boundaries

class Getlime:
    def __init__(self, model, imagepath):
        self.model = model
        self.imagepath = imagepath
        self.pill_transf = self.get_pil_transform()
        self.preprocess_transform = self.get_preprocess_transform()
        self.img = self.get_image()
   
    def get_image(self):
        with open(os.path.abspath(self.imagepath), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') 

    def get_pil_transform(self): 
        transf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224)
        ])    
        return transf

    def get_preprocess_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])     
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])    
        return transf    

    def batch_predict(self, images):
        self.model.eval()

        # Convert each image from numpy to PIL, then apply transformations
        pil_images = [Image.fromarray(img.astype('uint8'), 'RGB') for img in images]
        batch = torch.stack([self.preprocess_transform(self.pill_transf(pil_img)) for pil_img in pil_images], dim=0)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        batch = batch.to(device)
        
        # Get model predictions
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def get_bound_image(self):
        explainer = lime_image.LimeImageExplainer()

        # Explanation using Lime
        explanation = explainer.explain_instance(
            np.array(self.pill_transf(self.img)),
            self.batch_predict,  # Pass batch_predict as the classification function
            top_labels=5, 
            hide_color=0, 
            num_samples=1000
        )

        # Get image and mask for display
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        img_boundry = mark_boundaries(temp / 255.0, mask)
        
        return img_boundry, mask
        # plt.imshow(img_boundry)
        # plt.show()
                

# if __name__ == '__main__':
#     from model import load_model

#     # Load model and specify image path
#     model = load_model(r'C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\Final_project\resnet.pt') 
#     image_path = r"C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\archive\MexCulture142\images_val\Colonial_AcademiaDeBellasArtes_Queretaro_N_1.png"

#     # Instantiate Getlime and generate explanation
#     lime = Getlime(model, image_path)
#     lime.get_bound_image()
