from model import load_model, predict_image, extract_label_from_filename, data_transform, device
from visualization import represent_heatmap_overlaid
from rice import get_rice_heatmap
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from gradcam import Gradcam
from Fem import Feature_Explanation_method
from lime_explainer import Getlime
from evalmetrics import calculate_pcc


def get_gt_saliency(saliencypath, original_filename):
    base_name, extension = original_filename.rsplit('.', 1)
    last_underscore_index = base_name.rfind("_")
    second_last_underscore_index = base_name.rfind("_", 0, last_underscore_index)
    new_base_name = base_name[:second_last_underscore_index] + "_GFDM" + base_name[second_last_underscore_index:]
    new_filename = f"{new_base_name}.{extension}"
    frame_path_gt = os.path.join(saliencypath, new_filename)
    img_gt = Image.open(frame_path_gt)
    return img_gt


def main(modelpath, test_image_path, saliencypath, Xai_method, evalmet):
    model = load_model(modelpath)
    results = {"FILENAME": [], "GRADCAM": [], "FEM": [], "RICE": [], "LIME": []}
    test_image_dir = test_image_path
    correct_predictions = 0
    total_images = 0

    for img_file in os.listdir(test_image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(test_image_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            input_batch = data_transform(image).unsqueeze(0).to(device)
            ground_truth = extract_label_from_filename(img_file)
            predicted_class, confidence = predict_image(img_path, model)
            is_correct = (predicted_class == ground_truth)
            correct_predictions += int(is_correct)
            total_images += 1

            print(f"Image: {img_file} | Ground Truth: {ground_truth} | Predicted Class: {predicted_class} | "
                  f"Confidence: {confidence:.2f} | Correct: {is_correct}")

            targets = None
            target_layers = [model.layer4[-1].conv2]
            gtforeval = get_gt_saliency(saliencypath, img_file)
            gt_saliency_resized = gtforeval.resize((224, 224), Image.LANCZOS)
            gt_np_sal = np.array(gt_saliency_resized)
            gt_saliency_normalized = gt_np_sal / 255.0

            if Xai_method == "all" and evalmet == 'pcc':
                results["FILENAME"].append(img_file)

                # GradCAM
                gcam = Gradcam(model, target_layers)
                heatmap = gcam(input_batch, targets)
                if heatmap.shape != gt_saliency_normalized.shape:
                    heatmap = cv2.resize(heatmap, (224, 224))  # Resize if needed
                pcc = calculate_pcc(gt_saliency_normalized, heatmap)
                results["GRADCAM"].append(pcc)

                # FEM
                fem = Feature_Explanation_method(model, target_layers)
                heatmap = fem(input_batch)
                if heatmap.shape != gt_saliency_normalized.shape:
                    heatmap = cv2.resize(heatmap, (224, 224))
                pcc = calculate_pcc(gt_saliency_normalized, heatmap)
                results["FEM"].append(pcc)

                # RICE
                heatmap = get_rice_heatmap(model, input_batch)
                if heatmap.shape != gt_saliency_normalized.shape:
                    heatmap = cv2.resize(heatmap, (224, 224))
                pcc = calculate_pcc(gt_saliency_normalized, heatmap)
                results["RICE"].append(pcc)

                # LIME
                lime_x = Getlime(model, img_path)
                _, heatmap = lime_x.get_bound_image()
                if heatmap.shape != gt_saliency_normalized.shape:
                    heatmap = cv2.resize(heatmap, (224, 224))
                pcc = calculate_pcc(gt_saliency_normalized, heatmap)
                results["LIME"].append(pcc)

            print(f"The {img_file} results saved!")

    if Xai_method == "all":
        df = pd.DataFrame(results)
        df.to_csv(f"{evalmet}.csv", index=False)


if __name__ == '__main__':
    modelpath = r'C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\Final_project\resnet.pt'
    testimages = r"C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\archive\MexCulture142\images_val"
    saliencypath = r'C:\Users\rohin\Desktop\New folder (3)\DeepLearning in Computer Vision\archive\MexCulture142\gazefixationsdensitymaps'
    main(modelpath, testimages, saliencypath, "all", "pcc")
