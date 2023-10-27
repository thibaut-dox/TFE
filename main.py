import os
import numpy as np

import PIL
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.transforms.functional import to_pil_image

from torchvision.utils import save_image

from dataset_creation import imageTensors
from training import MyCNN, tuneHyperparam

# GradCAM implementation for the heatmaps
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.target_layer.register_forward_hook(self.save_feature_maps)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def __call__(self, x):
        self.model.zero_grad()  # Clear any previous gradients
        with torch.enable_grad():
            return self.model(x)


# Hook for taking gradients for the heatmaps
def backward_hook1(module, grad_input, grad_output):
    global gradients  # refers to the variable in the global scope
    gradients = grad_output


# Hook for taking the ouput for the heatmaps
def forward_hook1(module, args, output):
    global activations
    activations = output


# Global variable for the hooks
gradients = None
activations = None


# Create the heatmaps with specifiq target class (0=bugged, 1=clean,, 2=defect)
def heatmap(model, image, save_dir, nbr_heat, target_class):

    backward_hook = model.conv4.register_full_backward_hook(backward_hook1, prepend=False)
    forward_hook = model.conv4.register_forward_hook(forward_hook1, prepend=False)
    pred = model(image)

    if pred[0][target_class] == torch.max(pred):
        pred[0][target_class].backward()
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
        # weight the channels by corresponding gradients
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        heatmap = nn.functional.relu(heatmap)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        overlay = to_pil_image(heatmap.detach(), mode='F').resize((256,256), resample=PIL.Image.Resampling.BICUBIC)
        cmap = colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

        # Create a figure and plot the first image
        fig, ax = plt.subplots()
        ax.axis('off')  # removes the axis markers

        # First plot the original image
        ax.imshow(to_pil_image(image.squeeze(), mode='RGB'))
        ax.imshow(overlay, alpha=0.4, interpolation='nearest')

        #save the fig
        heatmap_path = os.path.join(save_dir, f"heatmap_test{nbr_heat}.jpg")
        plt.savefig(heatmap_path)

    backward_hook.remove()
    forward_hook.remove()


# Save images that are not correctly classified for given images
def misclassified(predictions, labels, images, nbr_mis, save_dir):
    for i in range(len(predictions)):
        if predictions[i] != labels[i]:
            image_label = labels[i]
            image_pred = predictions[i]
            filename = f"image{nbr_mis}_label{image_label}_pred{image_pred}.jpg"
            image_path = os.path.join(save_dir, filename)
            save_image(images[i], image_path)
            nbr_mis += 1


# Calculating the confusion matrix and other metric of the test set
def calculate_metrics(predictions, labels, nbr_labels, save_dir):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=list(range(nbr_labels)))

    # Save conf matrix
    class_labels = ['Bugged', 'Clean', 'Defect']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    conf_mat_path = os.path.join(save_dir, f"Confusion matrix.jpg")
    plt.savefig(conf_mat_path)


    # Calculate classification report
    report = classification_report(labels.cpu().numpy(), predictions.cpu().numpy(), labels=list(range(nbr_labels)), zero_division=0)
    print(report)


# To draw the roc curves of the test set
def calculate_roc(predictions, labels, nbr_labels, save_dir):
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(nbr_labels):
        fpr[i], tpr[i], _ = roc_curve(labels == i, predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 6))

    for i in range(nbr_labels):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc="lower right")
    roc_path = os.path.join(save_dir, f"ARoc curves.jpg")
    plt.savefig(roc_path)
    plt.close()


# Test the model : produce accuracy, heatmaps, misclassified
def testModel(model, testDS, nbr_labels):

    # To have a look on the misclassified images
    save_dir = 'C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/Misclassified_images'
    os.makedirs(save_dir, exist_ok=True)

    # Where to save the heatmaps
    heatmap_dir = 'C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/Heatmap'
    os.makedirs(heatmap_dir, exist_ok=True)

    test_loader = torch.utils.data.DataLoader(testDS, batch_size=batch_size, shuffle=False)
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


# Check if the folder of folder_path is empty
def is_empty_folder(folder_path):
    return not any(os.listdir(folder_path))


# Return the number of labels (the number of folder in folder_path)
def count_labels(folder_path):
    items = os.listdir(folder_path)
    folder_count = 0

    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_count += 1
    return folder_count


# Give prediction of the model for a unique image
def prediction(model, image):
    image = image.to(device)
    prediciton = model(image.unsqueeze())
    _, pred = torch.max(prediciton, dim=1)
    if pred.item() == 0:
        print("The model predict that the image is Bugged")
    elif pred.item() == 1:
        print("The model predict that the image is Clean")
    elif pred.item() == 2:
        print("The model predict that the image is Defect")


if __name__ == '__main__':

    # To know if we work on GPU or not
    disp = True
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if disp:
        print('The device used is : ' + str(device))

    # HERE : need to adapt the path correctly with they will be stored
    # Path to the stored images and where to save the preprocess dataset, for faster computing during test of the code
    directoryToProcess = "C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/PicturesSORTED"
    pathStored = "C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/ProcessedJob"

    # Parameters
    num_epochs = 20
    lr = 0.001
    image_size = 256
    nbr_channels = 16
    batch_size = 100

    # To not recreate dataset when we do test on the model
    if is_empty_folder(pathStored):
        imageTensors(directoryToProcess, pathStored, image_size)
    else:
        print("Took precedent dataset")

    # Retrieving the dataset
    train_dataset = torch.load(pathStored + "/trainDS.pt")
    val_dataset = torch.load(pathStored + "/valDS.pt")
    test_dataset = torch.load(pathStored + "/testDS.pt")
    nbr_labels = count_labels(directoryToProcess)

    # To not retrain a model when we do test on the model
    if not os.path.exists("model.pt"):
        trained_model = tuneHyperparam(train_dataset, val_dataset, lr, batch_size, nbr_labels, nbr_channels, image_size, device, num_epochs)
        trained_model.to(device)
        torch.save(trained_model.state_dict(), 'model.pt')
    else:
        trained_model = MyCNN(nbr_labels, nbr_channels, image_size)
        trained_model.to(device)
        trained_model.load_state_dict(torch.load('model.pt'))

    testModel(trained_model, test_dataset, nbr_labels)

