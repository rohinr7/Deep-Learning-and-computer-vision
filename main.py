"""
Author : Rohin Andy Ramesh
DLCV lab final XAI
University of bordeaux

"""


import argparse
from model import load_model ,predict_image ,extract_label_from_filename , data_transform ,device
from visualization import represent_heatmap_overlaid ,visualize_saliency
from rice import get_rice_heatmap
from utils_lab import get_gt_saliency,plot_overlaid_heatmaps , plot_original_and_saliency
import os 
from PIL import Image
from gradcam import Gradcam
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from Fem import Feature_Explanation_method
from lime_explainer import Getlime
from evalmetrics import calculate_pcc ,calculate_sim ,deletion_auc,insertion_auc



def main(modelpath, test_image_path, saliencypath, Xai_method, evalmet):
        """ 
        Xai : string please write which explanation to be viewed

        """
        model = load_model(modelpath)

        # Directory containing new images to test
        test_image_dir = test_image_path  # Change this path to your images directory
        limitcount = 0 
        # Initialize counters for accuracy calculation
        correct_predictions = 0
        total_images = 0

        # Loop through each image in the directory and perform inference
        for img_file in os.listdir(test_image_dir):  
            results = {"XAI": ["PCC", "SIM", "dELETION", "INSERTION"] ,
                "GRADCAM" : [],
                "FEM": [],
                "RICE" : [],
                "LIME": []   }  
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
                    plot_original_and_saliency(image, saliency_xai)

                elif Xai_method == "fem":
                    fem =  Feature_Explanation_method(model, target_layers)
                    heatmap= fem(input_batch)
                    saliency_xai = represent_heatmap_overlaid(image, heatmap, colormap="jet") 
                    plot_original_and_saliency(image, saliency_xai)

                elif Xai_method == "rice":
                    heatmap = get_rice_heatmap(model, input_batch)
                    saliency_xai = represent_heatmap_overlaid(image, heatmap, colormap="jet")
                    plot_original_and_saliency(image, saliency_xai)

                elif Xai_method == "lime":
                    lime_x = Getlime(model, img_path)
                    _ ,heatmap= lime_x.get_bound_image()  
                    saliency_xai = represent_heatmap_overlaid(image, heatmap, colormap="jet")
                    plot_original_and_saliency(image, saliency_xai)

                    sim = calculate_sim(gt_saliency_normalized,heatmap)
                    print(f"the value of the Sim is {sim}")

                    # print(f" the type of the hepmap is {type(heatmap)} and shape {heatmap.shape} max : {np.max(heatmap)}  min : {np.min(heatmap)}")
                elif Xai_method == "all":
                    if evalmet:

                        saliency_gt = get_gt_saliency(saliencypath,img_file)
                        saliencyft_overlaid = represent_heatmap_overlaid(image,saliency_gt,colormap="jet")

                        gcam = Gradcam(model, target_layers)
                        GDheatmap = gcam(input_batch, targets)
                        GDsaliency_ol = represent_heatmap_overlaid(image, GDheatmap, colormap="jet") 

                        fem =  Feature_Explanation_method(model, target_layers)
                        FEMheatmap= fem(input_batch)
                        FEMsaliency_ol = represent_heatmap_overlaid(image, FEMheatmap, colormap="jet")


                        RICheatmap = get_rice_heatmap(model, input_batch)
                        RICsaliency_ol = represent_heatmap_overlaid(image, RICheatmap, colormap="jet")

                        lime_x = Getlime(model, img_path)
                        _ ,LIMheatmap= lime_x.get_bound_image()
                        LIMEsaliency_ol = represent_heatmap_overlaid(image, LIMheatmap, colormap="jet")


                        ##############################################################
                        pcc = calculate_pcc(gt_saliency_normalized,GDheatmap)
                        results["GRADCAM"].append(pcc) 

                        pcc = calculate_pcc(gt_saliency_normalized,FEMheatmap)
                        results["FEM"].append(pcc)

                        pcc = calculate_pcc(gt_saliency_normalized,RICheatmap)
                        results["RICE"].append(pcc)

                        pcc = calculate_pcc(gt_saliency_normalized,LIMheatmap)
                        results["LIME"].append(pcc)

                        ############################################################## 
                        sim = calculate_sim(gt_saliency_normalized,GDheatmap)
                        results["GRADCAM"].append(sim) 

                        sim = calculate_sim(gt_saliency_normalized,FEMheatmap)
                        results["FEM"].append(sim) 

                        sim = calculate_sim(gt_saliency_normalized,RICheatmap)
                        results["RICE"].append(sim) 

                        sim = calculate_sim(gt_saliency_normalized,LIMheatmap)
                        results["LIME"].append(sim) 
                        ###############################################################
                        _, delscore = deletion_auc(model,image,input_batch,GDheatmap,800)
                        results["GRADCAM"].append(delscore) 

                        _, delscore = deletion_auc(model,image,input_batch,FEMheatmap,800)
                        results["FEM"].append(delscore) 

                        _, delscore = deletion_auc(model,image,input_batch,RICheatmap,800)
                        results["RICE"].append(delscore) 

                        _, delscore = deletion_auc(model,image,input_batch,LIMheatmap,800)
                        results["LIME"].append(delscore) 
                        ################################################################
                        _, insscore = insertion_auc(model,image,input_batch,GDheatmap,800)
                        results["GRADCAM"].append(insscore) 

                        _, insscore = insertion_auc(model,image,input_batch,FEMheatmap,800)
                        results["FEM"].append(insscore) 

                        _, insscore = insertion_auc(model,image,input_batch,RICheatmap,800)
                        results["RICE"].append(insscore) 

                        _, insscore = insertion_auc(model,image,input_batch,LIMheatmap,800)
                        results["LIME"].append(insscore) 


                        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
                        df.to_csv(f"{os.path.splitext(img_file)[0]}.csv", index=False)
                        print(df)

                        plot_overlaid_heatmaps(image, saliencyft_overlaid, GDsaliency_ol, FEMsaliency_ol, RICsaliency_ol, LIMEsaliency_ol)
                
# if __name__ == '__main__':
#     modelpath =  "Deep-Learning-and-computer-vision-main/resnet.pt"
#     testimages =   "/net/ens/DeepLearning/DLCV2024/MexCulture142/images_val"
#     saliencypath = "/net/ens/DeepLearning/DLCV2024/MexCulture142/gazefixationsdensitymaps"
#     main(modelpath,testimages,saliencypath,"all",True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference with optional parameters.")
    
    # Mandatory argument
    parser.add_argument("modelpath", type=str, help="Path to the model file")
    
    # Optional arguments with defaults
    parser.add_argument(
        "--testimages", 
        type=str, 
        default="/net/ens/DeepLearning/DLCV2024/MexCulture142/images_val", 
        help="Path to the test images directory"
    )
    parser.add_argument(
        "--saliencypath", 
        type=str, 
        default="/net/ens/DeepLearning/DLCV2024/MexCulture142/gazefixationsdensitymaps", 
        help="Path to the saliency maps directory"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="all", 
        help="Mode of operation (default: all)"
    )
    parser.add_argument(
        "--verbose", 
        type=bool, 
        default=True, 
        help="Verbose output (default: True)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args.modelpath, args.testimages, args.saliencypath, args.mode, args.verbose)

