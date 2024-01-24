# All the metrics used for evaluating how well the model works

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
import seaborn as sns
from matplotlib import colormaps
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import PIL
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import wandb


# Define global variables for hooks
gradients = None
activations = None

# Hook for taking gradients for the heatmaps
def backward_hook1(module, grad_input, grad_output):
    global gradients  # refers to the variable in the global scope
    gradients = grad_output


# Hook for taking the ouput for the heatmaps
def forward_hook1(module, args, output):
    global activations
    activations = output

# Function to create the heatmaps
def heatmap(model, image, save_dir, nbr_heat, nn_choice):
    global activations
    global gradients

    if nn_choice == "resnet":
        last_conv_layer = model.layer4[-1].conv2
    elif nn_choice == "mycnn":
        last_conv_layer = model.conv4
    elif nn_choice == "vgg":
        last_conv_layer = model.features[-1]
    elif nn_choice == "inception":
        last_conv_layer = model.Mixed_7a

        # Save the current requires_grad state for each parameter
    original_requires_grad = [param.requires_grad for param in model.parameters()]

    # Set requires_grad to True for all parameters
    for param in model.parameters():
        param.requires_grad = True


    backward_hook = last_conv_layer.register_full_backward_hook(backward_hook1, prepend=False)
    forward_hook = last_conv_layer.register_forward_hook(forward_hook1, prepend=False)
    pred = model(image)

    if pred > 0.5:
        pred.backward()
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

        # Restore the original requires_grad state for each parameter
        for param, requires_grad in zip(model.parameters(), original_requires_grad):
            param.requires_grad = requires_grad

        matplotlib.use('Agg')
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

# Function to calculate all the metrics and confusion matrix
def calculate_metrics(predictions, labels, save_dir):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())

    # Save conf matrix
    # CHANGER NOM EN FONCTION IF BUGGED OR NOT, DEFECT OR NOT
    class_labels = ['Negative', 'Positive']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    conf_mat_path = os.path.join(save_dir, "Confusion_matrix.jpg")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.savefig(conf_mat_path)
    plt.close()

    # Calculate classification report
    report = classification_report(labels.cpu().numpy(), predictions.cpu().numpy(), target_names=class_labels, zero_division=0)
    print(report)


# To draw the roc curves of the test set
def calculate_roc(predictions, labels, save_dir):
    fpr, tpr, _ = roc_curve(labels, predictions) # = positive = defect
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))

    plt.plot(fpr, tpr, lw=2, label=f'Class 1 (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc="lower right")
    roc_path = os.path.join(save_dir, f"ARoc curves.jpg")
    plt.savefig(roc_path)
    wandb.log({"Roc curves": wandb.Image(plt)})
    plt.close()

