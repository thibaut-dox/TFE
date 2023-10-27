import torch
import torch.nn as nn
import time
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter

# For the tensorboard
writer = SummaryWriter("runs/tfe")
writer_train = SummaryWriter("runs/train")
writer_val = SummaryWriter("runs/val")


# Implementation of the CNN
class MyCNN(nn.Module):
    def __init__(self, nbr_labels, nbr_channels, image_size):
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
        self.fc2 = nn.Linear(64, nbr_labels)
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


# Calculating class distribution
def calculate_class_distribution(dataset):
    print("Calculating class distribution")
    class_counts = {}
    for _, label in dataset:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    return class_counts


# Train the model and early stopping after 3 epoch where validation loss does not go lower as the lowest so far
def tuneHyperparam(trainDS, valDS, lr, batch_size, nbr_labels, nbr_channels, image_size, device, num_epochs):

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

    # Calculating class distribution for the weighted cross entropy loss
    class_distribution = calculate_class_distribution(trainDS)
    class_frequencies = torch.zeros(len(class_distribution))
    for label, count in class_distribution.items():
        class_frequencies[label] = count
    total_samples = class_frequencies.sum().item()
    class_weights = [total_samples / (len(class_frequencies) * freq) for freq in class_frequencies]

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    smallest_evallos = float('inf')
    start_time = time.time()
    stop = 0
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
    #writer_train.close()
    #writer_val.close()

    return model
