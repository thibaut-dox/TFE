import os
import torch
import pickle

import wandb


from dataset_creation import imageTensors
from training import tuneHyperparam, choose_NN
from testing import testModel


# Check if the folder of folder_path is empty
def is_empty_folder(folder_path):
    return not any(os.listdir(folder_path))


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

# Model pipeline for the sweeps
def model_pipeline():

    with wandb.init():
        config = wandb.config

        # To not recreate dataset when we do test on the model
        if is_empty_folder(pathStored):
            imageTensors(directoryToProcess, pathStored, config)
        else:
            print("Took precedent dataset")
        # Retrieving the dataset
        train_dataset = torch.load(pathStored + "/trainDS.pt")
        val_dataset = torch.load(pathStored + "/valDS.pt")
        test_dataset = torch.load(pathStored + "/testDS.pt")
        with open(pathStored + "/weight.pkl", "rb") as file:
            class_weights = pickle.load(file)
        print("Dataset loaded")



        if not os.path.exists("model.pt"):
            trained_model = tuneHyperparam(train_dataset, val_dataset, device, class_weights, config)
            trained_model.to(device)
            torch.save(trained_model.state_dict(), 'model.pt')
        else:
            print("Took precedent model")
            nbr_channels = config.nbr_channels
            image_size = config.image_size
            nn_choice = config.nn_choice
            _, _, trained_model = choose_NN(nn_choice, nbr_channels, image_size, device, val_dataset.transform, train_dataset.transform)
            trained_model.to(device)
            trained_model.load_state_dict(torch.load('model.pt'))
        
        testModel(trained_model, test_dataset, config, device)



if __name__ == '__main__':

    os.environ["WANDB__SERVICE_WAIT"] = "300"


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
    # path to store the class_weight for the weighted cross entropy
    path_weight = "C:/Users/thibo/Documents/T-bow/Unif/master2/TFE/ProcessedJob/weight.pkl"

    # Define sweep config
    sweep_configuration = {
        'method': 'grid',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
    }

    # adapt the sweep to what we want to test
    parameters_dict = {
        'batch_size': {
            'values': [256]
        },
        'epochs': {
            'values': [20]
        },
        'lr': {
            'values': [0.001]
        },
        'image_size': {
            'values': [256]
        },
        'nbr_channels': {
            'values': [8]
        },
        'nn_choice': {
            'values': ["mycnn"]
        },
        'excluded_job': {
            'values': ['JOB 11']
        },

    }
    sweep_configuration['parameters'] = parameters_dict
    # Initialize sweep by passing in config.
    # (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep_configuration, project="pre-trained")

    wandb.agent(sweep_id, model_pipeline, count=1)


