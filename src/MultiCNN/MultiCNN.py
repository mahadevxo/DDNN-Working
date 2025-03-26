import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Split image into tiles using PyTorch chunking
def split_into_tiles(image, num_tiles=4):
    """Splits an image into equal tiles."""
    B, C, H, W = image.shape
    h_chunks = torch.chunk(image, num_tiles, dim=2)  # Split along height
    tiles = [torch.chunk(h, num_tiles, dim=3) for h in h_chunks]  # Split along width
    tiles = [tile for row in tiles for tile in row]  # Flatten list
    return tiles 

# Feature extractor using ResNet18
class FeatureCNN(nn.Module):
    def __init__(self):
        super(FeatureCNN, self).__init__()
        resnet = models.vgg13(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers
    
    def forward(self, x):
        return self.feature_extractor(x)  # Shape: (B, 512, H/32, W/32)

# Pooling & Classification
class PoolingCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(PoolingCNN, self).__init__()
        self.conv = nn.Conv2d(8192, 1024, kernel_size=3, padding=1)  # Reduce depth
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv(x)  # Shape: (B, 1024, H, W)
        x = self.gap(x).squeeze(-1).squeeze(-1)  # Global Avg Pooling
        return self.fc(x)  # Shape: (B, num_classes)

# Multi-view CNN with improved tiling & feature extraction
class MultiCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(MultiCNN, self).__init__()
        self.num_tiles = 4  # Using 4 tiles instead of 16
        self.feature_cnns = nn.ModuleList([FeatureCNN() for _ in range(self.num_tiles)])
        self.pooling_cnn = PoolingCNN(num_classes)

    def forward(self, x):
        tiles = split_into_tiles(x, num_tiles=self.num_tiles)
        features = [cnn(tile) for cnn, tile in zip(self.feature_cnns, tiles)]
        combined_features = torch.cat(features, dim=1)  # Shape: (B, 8192, H, W)
        return self.pooling_cnn(combined_features)

# Training function with scheduler
def train_model(model, dataloader, criterion, optimizer, scheduler, device, num_epochs=20):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()  # Adjust learning rate
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100*correct/total:.2f}%")

# Main function with augmentations
def main():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    from SportsDataset import SportsDataset

    dataset_path = "sports.csv"
    train_dataset = SportsDataset(csv_file=dataset_path, root_dir="./", transform=transform, split="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiCNN(num_classes=len(train_dataset.data['class id'].unique()))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs

    train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=20)

if __name__ == "__main__":
    main()