# Test the model and calculate some metrics and heatmaps

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from training import choose_NN


from torch.utils.data import DataLoader

from tqdm import tqdm

from metrics import calculate_metrics, calculate_roc, heatmap, misclassified

# Test the model : produce accuracy, heatmaps, misclassified
def testModel(model, testDS, config, device):

    batch_size = config.batch_size

    # To have a look on the misclassified images
    #save_dir = 'C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/Misclassified_images'
    #os.makedirs(save_dir, exist_ok=True)

    # Where to save the heatmaps
    heatmap_dir = 'C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/Heatmap'
    os.makedirs(heatmap_dir, exist_ok=True)

    nbr_channels = config.nbr_channels
    image_size = config.image_size
    nn_choice = config.nn_choice

    new_tranform, _, _ = choose_NN(nn_choice, nbr_channels, image_size, device, testDS.transform, testDS.transform)
    testDS.transform = new_tranform

    test_loader = DataLoader(testDS, batch_size=batch_size, shuffle=False)
    nbr_heat = 0

    model.eval()

    # Creating the heatmaps, with the target score of the defect prediction (0=bugged, 1=clean,, 2=defect)
    for images, labels in tqdm(test_loader, desc= "Heatmaps"):
        images = images.to(device)
        for image, label in zip(images, labels):
            if label == 1:
                heatmap(model, image.unsqueeze(0), heatmap_dir, nbr_heat, nn_choice)
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
            threshold = 0.5
            predictions = (proba_prediction > threshold).float()

            all_true_labels.append(labels)
            all_predicted_labels.append(predictions)
            all_predicted_probs.append(proba_prediction)

            # Save the misclassified image
            #misclassified(predictions, labels, images, nbr_mis, save_dir)
            nbr_mis += len(predictions)

    # Combine predictions and labels for the entire dataset
    all_true_labels = torch.cat(all_true_labels)
    all_predicted_labels = torch.cat(all_predicted_labels)
    all_predicted_probs = torch.cat(all_predicted_probs)

    # Calculate accuracy metrics
    calculate_metrics(all_predicted_labels, all_true_labels, heatmap_dir)

    # Calculate ROC metrics
    all_true_labels = np.array(all_true_labels.cpu().numpy())
    all_predicted_probs = np.array(all_predicted_probs.cpu().numpy())
    calculate_roc(all_predicted_probs, all_true_labels, heatmap_dir)