from model import load_model ,predict_image ,extract_label_from_filename , data_transform ,device
from visualization import represent_heatmap_overlaid ,visualize_saliency
from rice import get_rice_heatmap
import os 
from PIL import Image
from gradcam import Gradcam
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from Fem import Feature_Explanation_method
from lime_explainer import Getlime
from evalmetrics import calculate_pcc ,calculate_sim


def get_gt_saliency(saliencypath, original_filename):
    # Split filename and extension
    base_name, extension = original_filename.rsplit('.', 1)  # Split only on the last dot
    # Find the second-to-last underscore (right before the last part)
    last_underscore_index = base_name.rfind("_")
    second_last_underscore_index = base_name.rfind("_", 0, last_underscore_index)
    # Insert _GFDM at the second-to-last underscore position
    new_base_name = base_name[:second_last_underscore_index] + "_GFDM" + base_name[second_last_underscore_index:]
    # Combine with extension for the new filename
    new_filename = f"{new_base_name}.{extension}"
    frame_path_gt = os.path.join(saliencypath ,new_filename)
    img_gt = Image.open(frame_path_gt)
    return img_gt

def main(modelpath,test_image_path, saliencypath, Xai_method, evalmet):
    """ 
    Xai : string please write which explanation to be viewed

    """
    model = load_model(modelpath)
    
    results = {"FILENAME": [] ,
               "GRADCAM" : [],
               "FEM": [],
               "RICE" : [],
               "LIME": []   }  

    # Directory containing new images to test
    test_image_dir = test_image_path  # Change this path to your images directory

    # Initialize counters for accuracy calculation
    correct_predictions = 0
    total_images = 0

    # Loop through each image in the directory and perform inference
    for img_file in os.listdir(test_image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(test_image_dir, img_file)

            # transformed input batch 
            image = Image.open(img_path).convert("RGB")
            input_batch = data_transform(image).unsqueeze(0).to(device)

            ground_truth = extract_label_from_filename(img_file)  
            # Predict class and get confidence
            predicted_class, confidence = predict_image(img_path, model)
            # Check if prediction is correct
            is_correct = (predicted_class == ground_truth)
            correct_predictions += int(is_correct)
            total_images += 1

            # Print result with ground truth
            print(f"Image: {img_file} | Ground Truth: {ground_truth} | Predicted Class: {predicted_class} | Confidence: {confidence:.2f} | Correct: {is_correct}")
            
            targets = None
            target_layers = [model.layer4[-1].conv2]

            gtforeval = get_gt_saliency(saliencypath,img_file)
            gt_saliency_resized = gtforeval.resize((224, 224), Image.LANCZOS)
            gt_np_sal = np.array(gt_saliency_resized)
            gt_saliency_normalized = gt_np_sal / 255.0

            if Xai_method == "gradcam":
                gcam = Gradcam(model, target_layers)
                heatmap = gcam(input_batch, targets)
                saliency_xai = represent_heatmap_overlaid(image, heatmap, colormap="jet")

            elif Xai_method == "fem":
                fem =  Feature_Explanation_method(model, target_layers)
                heatmap= fem(input_batch)
                saliency_xai = represent_heatmap_overlaid(image, heatmap, colormap="jet") 


            elif Xai_method == "rice":
                heatmap = get_rice_heatmap(model, input_batch)
                saliency_xai = represent_heatmap_overlaid(image, heatmap, colormap="jet")

            elif Xai_method == "lime":
                lime_x = Getlime(model, img_path)
                _ ,heatmap= lime_x.get_bound_image()  
                saliency_xai = represent_heatmap_overlaid(image, heatmap, colormap="jet")

                # print(f" the type of the hepmap is {type(heatmap)} and shape {heatmap.shape} max : {np.max(heatmap)}  min : {np.min(heatmap)}")
            elif Xai_method == "all":
                if evalmet == 'pcc' :
                    results["FILENAME"].append(img_file)

                    gcam = Gradcam(model, target_layers)
                    heatmap = gcam(input_batch, targets)
                    print(f"The type of the heat map is {type(heatmap)} and the shape is {heatmap.shape} max and min is {np.max(heatmap), np.min(heatmap)}")
                    print(f"The type of the  GT heatmap is {type(gt_saliency_normalized)} and the shape is {gt_saliency_normalized.shape} max and min is {np.max(gt_saliency_normalized), np.min(gt_saliency_normalized)}")

                    pcc = calculate_pcc(gt_saliency_normalized,heatmap)
                    results["GRADCAM"].append(pcc)

                    fem =  Feature_Explanation_method(model, target_layers)
                    heatmap= fem(input_batch)
                    pcc = calculate_pcc(gt_saliency_normalized,heatmap)
                    results["FEM"].append(pcc)

                    heatmap = get_rice_heatmap(model, input_batch)
                    pcc = calculate_pcc(gt_saliency_normalized,heatmap)
                    results["RICE"].append(pcc)

                    lime_x = Getlime(model, img_path)
                    _ ,heatmap= lime_x.get_bound_image()
                    results["LIME"].append(pcc)

            print(f"The {img_file} results saved!")
            # #saliency gt
            # saliency_gt = get_gt_saliency(saliencypath,img_file)
            # saliencyft_overlaid = represent_heatmap_overlaid(image,saliency_gt,colormap="jet")
            # # Overlay heatmap on the image
            # fig, ax = plt.subplots(1, 3, figsize=(12, 6))
            

            # #saliency ground truth

            # # Original image with overlayed heatmap
            # ax[0].imshow(saliencyft_overlaid)  # Adjust alpha for visibility
            # ax[0].axis('off')
            # ax[0].set_title(f"Gaze Fixation\nDensity maps ")
            
            # ax[1].imshow(saliency_xai)  # Adjust alpha for visibility
            # ax[1].axis('off')
            # ax[1].set_title(f"xai {Xai_method} : {predicted_class} (Conf: {confidence:.2f})")

            # # Display original image
            # ax[2].imshow(image)
            # ax[2].axis('off')
            # ax[2].set_title(f"Original Image\nGround Truth: {ground_truth}")

            # # Display the figure
            # plt.show()
    
    if Xai_method == "all":
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
        df.to_csv(f"{evalmet}.csv", index=False)


if __name__ == '__main__':
    modelpath =  r'C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\Final_project\resnet.pt'
    testimages =   r"C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\archive\MexCulture142\images_val"  
    saliencypath = r'C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\archive\MexCulture142\gazefixationsdensitymaps'
    main(modelpath,testimages,saliencypath,"all", "pcc")