import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image


def split_into_tiles(image, grid_ratios):
    """Splits an image into tiles based on given grid ratios."""
    B, C, H, W = image.shape
    h_splits = [int(H * r) for r in grid_ratios]
    w_splits = [int(W * r) for r in grid_ratios]

    # Ensure sum of splits does not exceed original H, W
    h_splits[-1] = H - sum(h_splits[:-1])
    w_splits[-1] = W - sum(w_splits[:-1])

    tiles = []
    start_h = 0
    for h_size in h_splits:
        start_w = 0
        for w_size in w_splits:
            tile = image[:, :, start_h:start_h + h_size, start_w:start_w + w_size]
            
            # Ensure all tiles are same size
            if tile.shape[2:] != (h_splits[0], w_splits[0]):  
                tile = torch.nn.functional.interpolate(tile, size=(h_splits[0], w_splits[0]), mode='bilinear', align_corners=False)

            tiles.append(tile)
            start_w += w_size
        start_h += h_size
    
    return tiles

class FeatureCNN(nn.Module):
    def __init__(self):
        super(FeatureCNN, self).__init__()
        self.feature_extractor = models.alexnet(weights=None).features
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
    
    def forward(self, x):
        return self.feature_extractor(x)

class PoolingCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(PoolingCNN, self).__init__()
        self.pooling_cnn = nn.Sequential(
            nn.Conv2d(2304, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.pooling_cnn(x)


class MultiCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(MultiCNN, self).__init__()
        self.feature_cnns = nn.ModuleList([FeatureCNN() for _ in range(16)])
        self.pooling_cnn = PoolingCNN(num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        tiles = split_into_tiles(x, [0.3, 0.3, 0.4])
        features = [cnn(tile) for cnn, tile in zip(self.feature_cnns, tiles)]
        combined_features = torch.cat(features, dim=1)  # Shape: (B, 4096, H, W)
        return self.pooling_cnn(combined_features)

# Function to save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Function to load model
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    print("Training the model...")
    model.to(device)
    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100*correct/total:.2f}%")


def main():
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    from SportsDataset import SportsDataset

    dataset_path = "sports.csv"
    train_dataset = SportsDataset(csv_file=dataset_path, root_dir="./", transform=transform, split="train")
    valid_dataset = SportsDataset(csv_file=dataset_path, root_dir="./", transform=transform, split="valid")
    test_dataset = SportsDataset(csv_file=dataset_path, root_dir="./", transform=transform, split="test")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiCNN(num_classes=len(train_dataset.data['class id'].unique()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Size of the dataset:", len(train_dataset))

    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

if __name__ == "__main__":
    main()
