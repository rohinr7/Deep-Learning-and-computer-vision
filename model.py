import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Classnames used in inference
classname = {
    0: 'Colonial',
    1: 'Modern',
    2: 'Prehispanic' 
}

# Define the transformation to be applied to input images
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path='resnet.pt'):
    # Load the model structure and weights
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

def predict_image(image_path, model):
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Apply transformations to the image
    image = data_transform(image).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = classname[predicted.item()]
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
        
    return predicted_class, confidence

def extract_label_from_filename(filename):
    # Extracts the ground truth label from the filename
    label_part = filename.split("_")[0]
    return label_part

def main_inference():
    # Load the trained model
    model = load_model(r'C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\Final_project\resnet.pt')
    
    # Directory containing new images to test
    test_image_dir = r"C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\archive\MexCulture142\images_val"  # Change this path to your images directory

    # Initialize counters for accuracy calculation
    correct_predictions = 0
    total_images = 0

    # Loop through each image in the directory and perform inference
    for img_file in os.listdir(test_image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(test_image_dir, img_file)
            
            # Extract ground truth from filename
            ground_truth = extract_label_from_filename(img_file)
            
            # Predict class and get confidence
            predicted_class, confidence = predict_image(img_path, model)
            
            # Check if prediction is correct
            is_correct = (predicted_class == ground_truth)
            correct_predictions += int(is_correct)
            total_images += 1

            # Print result with ground truth
            print(f"Image: {img_file} | Ground Truth: {ground_truth} | Predicted Class: {predicted_class} | Confidence: {confidence:.2f} | Correct: {is_correct}")

    # Calculate and print final accuracy
    accuracy = (correct_predictions / total_images) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")

# if __name__ == '__main__':
#     main_inference()
