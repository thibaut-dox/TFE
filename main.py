import os
import torch

from dataset_creation import imageTensors
from training import MyCNN, tuneHyperparam
from testing import testModel




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

    testModel(trained_model, test_dataset, nbr_labels, batch_size, device)

