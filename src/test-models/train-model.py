import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class TrainOtherModels():
    def __init__(self, save_model=False, num_classes=7, dataset_path='dataset'):
        model = models.vgg16(weights=None)
        model.classifier[-1] = nn.Linear(4096, num_classes)
        
        self.save_model = save_model
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.dataset_path = dataset_path
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
            
        
    def load_data(self, batch_size=32):
        class_names = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]
        label_map = {name: idx for idx, name in enumerate(class_names)}

        dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.transform)

        dataset.class_to_idx = label_map
        dataset.samples = [
            (path, label_map[os.path.basename(os.path.dirname(path))])
            for path, _ in dataset.samples
        ]
        dataset.targets = [label for _, label in dataset.samples]

        total_samples = len(dataset)
        train_size = int(0.8 * total_samples)
        val_size = total_samples - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def validate(self, val_loader):
        self.model.eval()
        total = 0
        correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        return accuracy

    def train(self, train_loader, num_epochs=100):
        self.model.train()
        try:
            for epoch in range(num_epochs):
                for images, labels in train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                epoch_accuracy = self.validate(train_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {epoch_accuracy:.2f}%, Loss: {loss.item():.4f}") # type: ignore
            
            if self.save_model:
                model_path = 'vgg16_trained.pth'
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
                
        except KeyboardInterrupt:
            print("Training interrupted. Saving the model state...")
            model_path = 'vgg16_partial.pth'
            torch.save(self.model.state_dict(), model_path)
            print(f"Partial model saved to {model_path}")
    
    def main(self, batch_size=32, num_epochs=100):
        train_loader, val_loader = self.load_data(batch_size)
        print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples.") # type: ignore
        self.train(train_loader, num_epochs)
        self.validate(val_loader)

if __name__ == "__main__":
    trainer = TrainOtherModels(save_model=True, num_classes=7, dataset_path='vehicle-classification')
    epochs = int(input("Enter number of epochs (default 100): ") or 100)
    trainer.main(batch_size=32, num_epochs=epochs)