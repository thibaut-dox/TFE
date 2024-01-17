# Use the loaded dataset to train a model on it

import wandb
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm





# Implementation of the CNN
class MyCNN(nn.Module):
    def __init__(self, nbr_channels, image_size):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, nbr_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nbr_channels)
        self.conv2 = nn.Conv2d(nbr_channels, nbr_channels*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(nbr_channels*2)
        self.conv3 = nn.Conv2d(nbr_channels*2, nbr_channels*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(nbr_channels*4)
        self.conv4 = nn.Conv2d(nbr_channels*4,  nbr_channels*8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(nbr_channels*8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(int(nbr_channels*8 * (image_size/16) * (image_size/16)), 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.nbr_channels = nbr_channels
        self.image_size = image_size

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

        x = x.view(-1, int(self.nbr_channels*8 * (self.image_size/16) * (self.image_size/16)))
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def choose_NN(nn_choice, nbr_channels, image_size, device, old_transform, old_normalize):
    if nn_choice == "mycnn":
        print("Model choice : MyCnn")
        model = MyCNN(nbr_channels, image_size)
        model.to(device)

        normalize = old_normalize
        new_transform = old_transform

    elif nn_choice == "resnet":
        print("Model choice : ResNet 18")
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.to(device)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        new_transform = transforms.Compose([
            transforms.CenterCrop(940),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    elif nn_choice == "vgg":
        print("Model choice : VGG 16")
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)
        model.to(device)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        new_transform = transforms.Compose([
            transforms.CenterCrop(940),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    elif nn_choice == "inception":
        print("Model choice : Inception v3")
        model = models.inception_v3(pretrained=True)
        for name, param in model.named_parameters():
            if "fc" not in name:  # excluding the final classifier
                param.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.to(device)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        new_transform = transforms.Compose([
            transforms.CenterCrop(940),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            normalize,
        ])


    return new_transform, normalize, model


# Train the model and early stopping after 3 epoch where validation loss does not go lower as the lowest so far
def tuneHyperparam(trainDS, valDS, device, class_weights, config):

    lr = config.lr
    batch_size = config.batch_size
    nbr_channels = config.nbr_channels
    image_size = config.image_size
    num_epochs = config.epochs
    nn_choice = config.nn_choice
    class_weights = torch.Tensor(class_weights).to(device)

    new_transform, normalize, model = choose_NN(nn_choice, nbr_channels, image_size, device, valDS.transform,
                                                trainDS.transform)

    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
        normalize,
    ])

    trainDS.transform = normalize
    trainDS.augmented_transform = augmentation_transform
    valDS.transform = new_transform

    train_loader = torch.utils.data.DataLoader(trainDS, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valDS, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight= class_weights.min() / class_weights.max())
    optimizer = optim.Adam(model.parameters(), lr=lr)

    wandb.watch(model, criterion, log="all", log_freq=1)
    print("Start training")
    smallest_evallos = float('inf')
    start_time = time.time()
    stop = 0
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"running_loss": loss.item()})
        wandb.log({"train_loss": running_loss, "epoch": epoch})

        model.eval()
        with torch.no_grad():
            evalloss = 0.0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                evalloss += loss.item()
            wandb.log({"val_loss" : evalloss, "epoch": epoch})
            print("Training loss : ", running_loss, " and Validation loss : ", evalloss)
        if smallest_evallos < evalloss:
            stop += 1
        else:
            smallest_evallos = evalloss
            stop = 0
        if stop == 3:
            print("3 consecutive times validation loss growing")
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds (training and validation)")


    return model
