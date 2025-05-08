import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        
        # Print class mapping for debugging
        print(f"Class mapping for {root_dir}:")
        for cls, idx in self.class_to_idx.items():
            print(f"{cls}: {idx}")
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
        
        print(f"Total images in {root_dir}: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a random image and label if there's an error
            return torch.zeros(3, 224, 224), 0

def get_transforms():
    """
    Get data transforms for training and validation
    Returns:
        train_transform: Transformations for training data
        val_transform: Transformations for validation data
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders(train_dir, val_dir, test_dir, batch_size=32):
    """
    Create data loaders for training, validation, and testing
    Args:
        train_dir (str): Path to training data
        val_dir (str): Path to validation data
        test_dir (str): Path to test data
        batch_size (int): Batch size for data loaders
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(train_dir, transform=train_transform)
    val_dataset = PlantDiseaseDataset(val_dir, transform=val_transform)
    test_dataset = PlantDiseaseDataset(test_dir, transform=val_transform)
    
    # Print dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model performance on a dataset
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
    Returns:
        accuracy: Model accuracy
        loss: Average loss
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss 