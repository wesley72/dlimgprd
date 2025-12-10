from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_loaders(batch_size=32, val_split=0.2):
    # Augmentation for training set
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),          # flip images left/right
        transforms.RandomRotation(10),              # rotate up to ±10 degrees
        transforms.ColorJitter(brightness=0.2, 
                               contrast=0.2, 
                               saturation=0.2, 
                               hue=0.1),           # vary color slightly
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # Validation & test transforms (no augmentation, just normalization)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # Load full dataset from 'data/train' with augmentation
    full_dataset = datasets.ImageFolder(root="data/train", transform=train_transform)

    # Split into train and validation sets
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override val_dataset transform (no augmentation)
    val_dataset.dataset.transform = eval_transform

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Test dataset loader (from 'data/test') with eval transform
    test_dataset = datasets.ImageFolder(root="data/test", transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader