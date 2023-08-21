# For going through files folder and be able to exit code without going through all of it
import os
import numpy as np
import time
import random
import PIL
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import cv2
from tqdm import tqdm
from torchviz import make_dot

# CNN and data related
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter("runs/tfe")
#writer_train = SummaryWriter("runs/train")
#writer_val = SummaryWriter("runs/val")



class MyCNN(nn.Module):
    def __init__(self, nbr_labels, nbr_channels, image_size):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, nbr_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nbr_channels)
        self.conv2 = nn.Conv2d(nbr_channels, nbr_channels*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(nbr_channels*2)
        self.conv3 = nn.Conv2d(nbr_channels*2, nbr_channels*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(nbr_channels*4)
        self.conv4 = nn.Conv2d(nbr_channels*4, nbr_channels*8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(nbr_channels*8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(int(nbr_channels*8 * (image_size/16) * (image_size/16)), 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, nbr_labels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, int(nbr_channels*8 * (image_size/16) * (image_size/16)))
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

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

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augmented_images, augmented_labels):
        self.base_dataset = base_dataset
        self.augmented_images = augmented_images
        self.augmented_labels = augmented_labels

    def __getitem__(self, index):
        if index < len(self.base_dataset):
            return self.base_dataset[index]
        else:
            new_index = index - len(self.base_dataset)
            return self.augmented_images[new_index], self.augmented_labels[new_index]

    def __len__(self):
        return len(self.base_dataset) + len(self.augmented_images)


def calculate_class_distribution(dataset):
    print("Calculating class distribution")
    class_counts = {}
    for _, label in dataset:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    return class_counts

# Preprocess from path : resize, grayscale and normalize
# Save the tensors to path2
def imageTensors(path_train, path_save, path_test, test, separated, image_size, centercrop):

    # Conditionally modify the transformation pipeline
    if centercrop:
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3), # Assure that it is in grayscale
            transforms.CenterCrop(940),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor()  # Convert images to tensors
        ])
    else:
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3), # Assure that it is in grayscale
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor()  # Convert images to tensors
        ])


    imageDS = ImageFolder(path_train, transform=transform)
    testDS = ImageFolder(path_test, transform=transform)
    # A subset of the dataset to test the code
    if test:
        # Define the size of the subset
        subset_size = 500
        imageDS, restDS = torch.utils.data.random_split(imageDS, [subset_size, len(imageDS)-subset_size])

    if separated:
        train_prop = 0.8
        train_size = int(train_prop * len(imageDS))
        val_size = len(imageDS) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(imageDS, [train_size, val_size])
        test_dataset = Subset(testDS, range(len(testDS)))
    else:
        imageDS = ConcatDataset([imageDS, testDS])
        train_prop = 0.7
        val_prop = 0.2
        train_size = int(train_prop * len(imageDS))
        val_size = int(val_prop * len(imageDS))
        test_size = len(imageDS) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(imageDS, [train_size, val_size, test_size])
    print("Initial size : ", len(train_dataset))
    trainloader = DataLoader(train_dataset, batch_size=200, shuffle=True)

    # Define a list of possible transformations
    possible_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
    ]

    # Define the maximum number of transformations to apply
    max_num_transforms = len(possible_transforms)

    # Apply random transformations to each image
    new_images = []
    new_labels = []
    images_tensor = []
    for batch, labels in trainloader:
        for img, label in zip(batch, labels):
            num_transforms = random.randint(1, max_num_transforms)  # Generate a random number of transformations
            chosen_transforms = random.sample(possible_transforms, num_transforms)  # Randomly select the transformations
            augmentation_transform = transforms.Compose(chosen_transforms)
            augmented_image = augmentation_transform(img)
            images_tensor.append(img)
            images_tensor.append(augmented_image)
            new_images.append(augmented_image)
            new_labels.append(label.item())
    tensor_images = torch.stack(images_tensor, dim=0)
    print("Final length of the train : ", len(tensor_images))

    mean = torch.mean(tensor_images, dim=[0, 2, 3])
    std = torch.std(tensor_images, dim=[0, 2, 3])

    train_dataset = AugmentedDataset(train_dataset, new_images, new_labels)

    torch.save(train_dataset, path_save + "/trainDS.pt")
    torch.save(val_dataset, path_save + "/valDS.pt")
    torch.save(test_dataset, path_save + "/testDS.pt")
    torch.save(mean, path_save + "/mean.pt")
    torch.save(std, path_save + "/std.pt")


# Train the model and stop when it start to overfit
def tuneHyperparam(trainDS, valDS, lr, batch_size, mean, std, image_size):

    normalize = transforms.Normalize(mean, std)
    trainDS.transform = normalize
    valDS.transform = normalize

    train_loader = torch.utils.data.DataLoader(trainDS, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valDS, batch_size=batch_size, shuffle=True)


    # Just to show an example of the data in the tensorboard
    examples = iter(train_loader)
    example_images, example_labels = next(examples)
    img_grid = make_grid(example_images)
    #writer.add_image('Example of the images', img_grid)
    #writer.close()
    model = MyCNN(nbr_labels, nbr_channels, image_size)
    model.to(device)

    # Assuming you have a dataset 'my_dataset'
    class_distribution = calculate_class_distribution(train_dataset)
    class_frequencies = torch.zeros(len(class_distribution))
    for label, count in class_distribution.items():
        class_frequencies[label] = count

    total_samples = class_frequencies.sum().item()

    class_weights = [total_samples / (len(class_frequencies) * freq) for freq in class_frequencies]

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    last_running_loss = float('inf')
    last_evalloss = float('inf')
    start_time = time.time()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #writer_train.add_scalar('Training loss vs Validation loss', loss.item(), epoch * len(train_loader) + i)
            writer_runs.add_scalar('Training loss vs Validation loss', loss.item(), epoch * len(train_loader) + i)

        model.eval()
        with torch.no_grad():
            evalloss = 0.0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                evalloss += loss.item()
            #writer_val.add_scalar('Training loss vs Validation loss', evalloss/len(val_loader), epoch*len(train_loader))
            writer_runs.add_scalar('Training loss vs Validation loss', evalloss/len(val_loader), epoch*len(train_loader))
            print("Training loss : ", running_loss, " and Validation loss : ", evalloss)
        if last_evalloss < evalloss and last_running_loss > running_loss:
            print("Finished training before last epoch")
            #break
        else:
            last_evalloss = evalloss
            last_running_loss = running_loss

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds (training and validation)")
    #writer_train.close()
    #writer_val.close()
    writer_runs.close()
    return model


def backward_hook1(module, grad_input, grad_output):
    global gradients  # refers to the variable in the global scope
    gradients = grad_output


def forward_hook1(module, args, output):
    global activations
    activations = output


gradients = None
activations = None


def heatmap(model, image, save_dir, nbr_heat, target_class):

    backward_hook = model.conv4.register_full_backward_hook(backward_hook1, prepend=False)
    forward_hook = model.conv4.register_forward_hook(forward_hook1, prepend=False)
    pred = model(image)
    if pred[2] == torch.max(pred):
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
def testModel(model, testDS, mean, std, nbr_labels):

    normalize = transforms.Normalize(mean, std)
    testDS.transform = normalize

    save_dir = 'C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/Misclassified_images'
    os.makedirs(save_dir, exist_ok=True)

    heatmap_dir = 'C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/Heatmap'
    os.makedirs(heatmap_dir, exist_ok=True)

    test_loader = torch.utils.data.DataLoader(testDS, batch_size=batch_size, shuffle=False)
    nbr_heat = 0
    model.eval()

    for images, labels in tqdm(test_loader, desc= "Heatmaps"):
        images = images.to(device)
        for image, label in zip(images, labels):
            if label == 2:
                heatmap(model, image.unsqueeze(0), heatmap_dir, nbr_heat)
                nbr_heat += 1
                plt.close()
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probs = []

    with torch.no_grad():
        correct_count = 0
        total_count = 0
        nbr_mis = 0
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            proba_prediction = model(images)
            probabilities, predictions = torch.max(proba_prediction, dim=1)

            # Overall accuracy
            correct_count += torch.sum(predictions == labels).item()
            total_count += len(labels)

            all_true_labels.append(labels)
            all_predicted_labels.append(predictions)
            all_predicted_probs.append(proba_prediction)

            # Save the misclassified image
            misclassified(predictions, labels, images, nbr_mis, save_dir)
            nbr_mis += len(predictions)

    # Calculate per-class accuracy
    accuracy = correct_count/total_count
    print("Total Accuracy:", accuracy)

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


if __name__ == '__main__':

    # TEST
    test = False
    if test:
        print("This is a test, we use a smaller dataset")
    else:
        print("This is with all the data")

    separated = False
    if separated:
        print("Image from test set are forms that the CNN never saw")

    # Debug
    disp = True
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if disp:
        print('The device used is : ' + str(device))

    # To go search the images at the right file, and save the dataset of the processed images on another file,
    # to retrieve it faster
    directoryToProcess = "C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/PicturesSORTED"
    pathStored = "C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/ProcessedJob"

    # If we use test images with forms never saw by the cnn
    path_test = "C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/Test_images"
    """
    if is_empty_folder(pathStored):
        print("Creating new dataset")
        imageTensors(directoryToProcess, pathStored, path_test, test, separated, image_size, centercrop)
        print("New dataset created")
    else:
        print("WARNING : Took precedent dataset")
    """
    """
    if not os.path.exists("model.pt"):
        print("Creating model")
        trained_model = tuneHyperparam(train_dataset, val_dataset, lr, batch_size, mean, std)
        trained_model.to(device)
        torch.save(trained_model.state_dict(), 'model.pt')
    else:
        trained_model = MyCNN(nbr_labels, nbr_channels)
        trained_model.to(device)
        trained_model.load_state_dict(torch.load('model.pt'))
    """
    num_random = 10
    num_epochs = 15
    lr = 0.001
    image_size = 128
    nbr_channels = 8
    batch_size = 100
    centercrop = False
    log_dir = 'runs/'

    print(f"Training with lr={lr}, image_size={image_size}, nbr_channels={nbr_channels}, batch_size={batch_size}, centercrop={centercrop}")
    if is_empty_folder(pathStored):
        print("Took precedent dataset")
        imageTensors(directoryToProcess, pathStored, path_test, test, separated, image_size, centercrop)
    train_dataset = torch.load(pathStored + "/trainDS.pt")
    val_dataset = torch.load(pathStored + "/valDS.pt")
    test_dataset = torch.load(pathStored + "/testDS.pt")
    mean = torch.load(pathStored + "/mean.pt")
    std = torch.load(pathStored + "/std.pt")
    nbr_labels = count_labels(directoryToProcess)

    # Create a new directory for this run's logs
    log_dir_run = os.path.join(log_dir, f'run_0')
    writer_runs = SummaryWriter(log_dir_run)
    # Log hyperparameters for this run
    writer_runs.add_scalar('Hyperparameters/lr', lr, 0)
    writer_runs.add_scalar('Hyperparameters/image_size', image_size, 0)
    writer_runs.add_scalar('Hyperparameters/nbr_channels', nbr_channels, 0)
    writer_runs.add_scalar('Hyperparameters/batch_size', batch_size, 0)
    trained_model = tuneHyperparam(train_dataset, val_dataset, lr, batch_size, mean, std, image_size)
    trained_model.to(device)
    testModel(trained_model, test_dataset, mean, std, nbr_labels)




