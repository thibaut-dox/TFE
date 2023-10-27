import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# To concatenate train dataset with new images
class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augmented_images, augmented_labels, transform):
        self.base_dataset = base_dataset
        self.augmented_images = augmented_images
        self.augmented_labels = augmented_labels
        self.transform = transform

    def __getitem__(self, index):
        if index < len(self.base_dataset):
            img, label = self.base_dataset[index]
            img = self.transform(img)
            return img, label
        else:
            new_index = index - len(self.base_dataset)
            img = self.augmented_images[new_index]
            img = self.transform(img)
            return img, self.augmented_labels[new_index]

    def __len__(self):
        return len(self.base_dataset) + len(self.augmented_images)


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


# Creating and preprocess datset : resize, centercrop and normalization
def imageTensors(path_train, path_save, image_size):

    transform = transforms.Compose([
        transforms.CenterCrop(940),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor()
    ])

    # Define the root directory and the job folder to exclude
    root_dir = path_train
    excluded_job_folder = 'JOB 7'

    file_list = []
    test_file_list = []

    # Populate the file list for training and validation sets, excluding excluded_job_folder
    for label, class_name in enumerate(['Bugged', 'Clean', 'Defect']):
        class_path = os.path.join(root_dir, class_name)
        for job_folder in os.listdir(class_path):
            if job_folder != excluded_job_folder:
                job_path = os.path.join(class_path, job_folder)
                if os.path.isdir(job_path):
                    for filename in os.listdir(job_path):
                        if filename.endswith('.jpg') or filename.endswith('.png'):
                            file_list.append((os.path.join(job_path, filename), label))
            else:
                job_path = os.path.join(class_path, job_folder)
                if os.path.isdir(job_path):
                    for filename in os.listdir(job_path):
                        if filename.endswith('.jpg') or filename.endswith('.png'):
                            test_file_list.append((os.path.join(job_path, filename), label))

    # Split the dataset into training and validation sets
    train_data, val_data = train_test_split(file_list, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = CustomDataset(train_data, transform=transform)
    val_dataset = CustomDataset(val_data, transform=transform)
    test_dataset = CustomDataset(test_file_list, transform=transform)

    # Define a list of possible transformations
    possible_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
    ]

    # Define the maximum number of transformations to apply
    max_num_transforms = len(possible_transforms)

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=True)

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

    normalize = transforms.Normalize(mean, std)
    train_dataset = AugmentedDataset(train_dataset, new_images, new_labels, normalize)

    new_transform = transforms.Compose([
        transform,
        normalize
    ])
    val_dataset.transform = new_transform
    test_dataset.transform = new_transform

    torch.save(train_dataset, path_save + "/trainDS.pt")
    torch.save(val_dataset, path_save + "/valDS.pt")
    torch.save(test_dataset, path_save + "/testDS.pt")