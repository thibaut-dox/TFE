import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tqdm import tqdm

from metrics import calculate_metrics, calculate_roc, heatmap, misclassified

# Test the model : produce accuracy, heatmaps, misclassified
def testModel(model, testDS, nbr_labels, batch_size, device):

    # To have a look on the misclassified images
    save_dir = 'C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/Misclassified_images'
    os.makedirs(save_dir, exist_ok=True)

    # Where to save the heatmaps
    heatmap_dir = 'C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/Heatmap'
    os.makedirs(heatmap_dir, exist_ok=True)

    test_loader = DataLoader(testDS, batch_size=batch_size, shuffle=False)
    nbr_heat = 0
    model.eval()

    # Creating the heatmaps, with the target score of the defect prediction (0=bugged, 1=clean,, 2=defect)
    for images, labels in tqdm(test_loader, desc= "Heatmaps"):
        images = images.to(device)
        for image, label in zip(images, labels):
            if label == 2:
                heatmap(model, image.unsqueeze(0), heatmap_dir, nbr_heat, 2)
                nbr_heat += 1
                plt.close()
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probs = []

    # Calculating the metrics
    with torch.no_grad():
        nbr_mis = 0
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            proba_prediction = model(images)
            probabilities, predictions = torch.max(proba_prediction, dim=1)

            all_true_labels.append(labels)
            all_predicted_labels.append(predictions)
            all_predicted_probs.append(proba_prediction)

            # Save the misclassified image
            misclassified(predictions, labels, images, nbr_mis, save_dir)
            nbr_mis += len(predictions)

    # Combine predictions and labels for the entire dataset
    all_true_labels = torch.cat(all_true_labels)
    all_predicted_labels = torch.cat(all_predicted_labels)
    all_predicted_probs = torch.cat(all_predicted_probs)
    all_predicted_probs = nn.functional.softmax(all_predicted_probs, dim=1)

    # Calculate accuracy metrics
    calculate_metrics(all_predicted_labels, all_true_labels, nbr_labels, heatmap_dir)

    # Calculate ROC metrics
    all_true_labels = np.array(all_true_labels.cpu().numpy())
    all_predicted_probs = np.array(all_predicted_probs.cpu().numpy())
    calculate_roc(all_predicted_probs, all_true_labels, nbr_labels, heatmap_dir)