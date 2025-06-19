import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os

class TrainPlaces365:
    def __init__(self, save_model=False, dataset_path='places365', num_classes=365, patience=5):
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(4096, num_classes)

        self.model = model
        self.save_model = save_model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.1, patience=2, verbose=True
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(self.device)
        self.dataset_path = dataset_path
        self.patience = patience

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self, batch_size=64):
        train_dir = os.path.join(self.dataset_path, 'train')
        val_dir = os.path.join(self.dataset_path, 'val')

        train_dataset = datasets.ImageFolder(root=train_dir, transform=self.transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=self.transform)

        print("Found", len(train_dataset.classes), "classes.")
        print("Train samples:", len(train_dataset), "Val samples:", len(val_dataset))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader

    def evaluate(self, loader):
        # sourcery skip: inline-immediately-returned-variable
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        return acc

    def train(self, train_loader, val_loader, num_epochs=50):
        best_val_acc = 0.0
        epochs_without_improvement = 0

        try:
            for epoch in range(num_epochs):
                self.model.train()
                running_loss = 0.0

                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                val_acc = self.evaluate(val_loader)
                self.scheduler.step(val_acc)

                print(f"Epoch [{epoch+1}/{num_epochs}] - Val Accuracy: {val_acc:.2f}%, Loss: {avg_loss:.4f}")

                # Early stopping check
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_without_improvement = 0

                    if self.save_model:
                        torch.save(self.model.state_dict(), 'vgg16_places365_best.pth')
                        print("Best model saved.")
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement} epoch(s).")

                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            torch.save(self.model.state_dict(), 'vgg16_places365_partial.pth')

    def main(self, batch_size=64, num_epochs=50):
        train_loader, val_loader = self.load_data(batch_size)
        self.train(train_loader, val_loader, num_epochs)
        final_val_acc = self.evaluate(val_loader)
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")

if __name__ == "__main__":
    trainer = TrainPlaces365(save_model=True, dataset_path='places365', num_classes=365, patience=5)

    while True:
        try:
            epochs = int(input("Epochs? (default 50): ") or 50)
        except ValueError:
            epochs = 50
        trainer.main(batch_size=32, num_epochs=epochs)

        cont = input("Train more? (y/n): ").strip().lower()
        if cont != 'y':
            break