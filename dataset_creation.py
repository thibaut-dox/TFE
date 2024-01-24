# Create the dataset, preprocess it and save it for future uses

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle



# To concatenate train dataset with new images
class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augmented_dataset, transform, augmented_transform):
        self.base_dataset = base_dataset
        self.augmented_dataset = augmented_dataset
        self.transform = transform
        self.augmented_transform = augmented_transform

    def __getitem__(self, index):
        if index < len(self.base_dataset):
            img, label = self.base_dataset[index]
            img = self.transform(img)
            return img, label
        else:
            new_index = index - len(self.base_dataset)
            img, label = self.augmented_dataset[new_index]
            img = self.augmented_transform(img)
            return img, label

    def __len__(self):
        return len(self.base_dataset) + len(self.augmented_dataset)


# To created dataset from all the different jobs
class CustomDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


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


# Creating and preprocess datset : resize, centercrop and normalization
def imageTensors(path_train, path_save, config):

    image_size = config.image_size
    excluded_job = config.excluded_job


    transform = transforms.Compose([
        transforms.CenterCrop(940),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor()
    ])

    # Define the root directory and the job folder to exclude
    root_dir = path_train
    excluded_job_folder = excluded_job

    file_list = []
    test_file_list = []

    print("Creating dataset..")
    # Populate the file list for training and validation sets, excluding excluded_job_folder, here for clean and defect images
    for label, class_name in enumerate(['Clean', 'Defect']):
        class_path = os.path.join(root_dir, class_name)

        for job_folder in os.listdir(class_path):
            if job_folder != excluded_job_folder:
                job_path = os.path.join(class_path, job_folder)
                if os.path.isdir(job_path):
                    for filename in os.listdir(job_path):
                        if filename.endswith('.jpg') or filename.endswith('.png'):
                            file_list.append((os.path.join(job_path, filename), label))
            else:
                # Add files to the test set
                job_path = os.path.join(class_path, job_folder)
                if os.path.isdir(job_path):
                    for filename in os.listdir(job_path):
                        if filename.endswith('.jpg') or filename.endswith('.png'):
                            test_file_list.append((os.path.join(job_path, filename), label))

    # Split the dataset into training and validation sets
    train_data, val_data = train_test_split(file_list, test_size=0.2, random_state=42)
    print("Dataset separated into train, valid and test")
    # Create datasets
    train_dataset = CustomDataset(train_data, transform=transform)
    val_dataset = CustomDataset(val_data, transform=transform)
    test_dataset = CustomDataset(test_file_list, transform=transform)

    # Define a list of all the transformations (choosing at random not possible if we want to use them to normalize
    # and still store only the path to the image
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
    ])

    trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    print("Start calculating mean and std")
    images_tensor = []
    for batch, labels in trainloader:
        for img, label in zip(batch, labels):
            augmented_image = augmentation_transform(img)
            images_tensor.append(img)
            images_tensor.append(augmented_image)
    tensor_images = torch.stack(images_tensor, dim=0)
    print("Calculation finished. Final length of the train : ", len(tensor_images))

    mean = torch.mean(tensor_images, dim=[0, 2, 3])
    std = torch.std(tensor_images, dim=[0, 2, 3])

    normalize = transforms.Normalize(mean, std)
    augmentation_transform = transforms.Compose([
        augmentation_transform,
        normalize
    ])

    train_dataset = AugmentedDataset(train_dataset, train_dataset, normalize, augmentation_transform)
    new_transform = transforms.Compose([
        transform,
        normalize
    ])

    val_dataset.transform = new_transform
    test_dataset.transform = new_transform
    

    torch.save(train_dataset, path_save + "/trainDS.pt")
    torch.save(val_dataset, path_save + "/valDS.pt")
    torch.save(test_dataset, path_save + "/testDS.pt")


    # Calculating class distribution for the weighted cross entropy loss
    class_distribution = calculate_class_distribution(train_dataset)
    class_frequencies = torch.zeros(len(class_distribution))
    for label, count in class_distribution.items():
        class_frequencies[label] = count
    total_samples = class_frequencies.sum().item()
    class_weights = torch.tensor([total_samples / class_frequencies[0], total_samples / class_frequencies[1]])

    with open(path_save + "/weight.pkl", "wb") as file:
        pickle.dump(class_weights, file)
    print("Dataset and distribution calculated")