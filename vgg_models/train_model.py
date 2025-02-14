from cv2 import transform
import torch.nn as nn
from torchvision import models, transforms
import torch
from torchvision import datasets

class VGG16Sigmoid(nn.Module):
    def __init__(self):
        super(VGG16Sigmoid, self).__init__()
        self.model = models.vgg16(weights=None)
        self.model.classifier[-1] = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
        self.features = self.model.features
        self.classifier = self.model.classifier
    
    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)
    
if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.Resize((255,255)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
    

    dataset = datasets.ImageFolder('cat-dogs-dataset/training_set/training_set', transform=transform)
    eval_dataset = datasets.ImageFolder('cat-dogs-dataset/test_set/test_set', transform=transform)
    target_to_class = {
        0: 'cat',
        1: 'dog'
    }
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    train_size = int(0.8 * len(dataset))  
    test_size = len(dataset) - train_size
    
    print(f"Train Size: {train_size} - Test Size: {test_size}")

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=True, num_workers=4)


    def print_label_distribution(loader, name):
        print(f'\n{name} Data')
        label_counts = {}
        
        for images, labels in loader:
            for label in labels:
                label_name = target_to_class[int(label.item())]
                label_counts[label_name] = label_counts.get(label_name, 0) + 1

        for label_name, count in label_counts.items():
            print(f"{label_name}: {count} images")
            
    # print_label_distribution(train_loader, 'Train')
    # print_label_distribution(test_loader, 'Test')
    # print_label_distribution(eval_loader, 'Eval')        



    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    model = VGG16Sigmoid().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


    num_epochs = int(input('Enter number of epochs: '))
    loss_list = []
    accuracy_list = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = correct = total = 0
        
        print(f"--------------------------Epoch {epoch+1}/{num_epochs}--------------------------")

        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.float().to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total += labels.size(0)
            
            predicted = (outputs > 0.5).float()
            correct += (predicted==labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)   
        epoch_accuracy = (correct / total)*100
        
        loss_list.append(epoch_loss)
        accuracy_list.append(epoch_accuracy)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")
        
        
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            
        eval_accuracy = (correct / total)*100
        print(f"Eval Accuracy: {eval_accuracy:.2f}%")
        
    torch.save(model, 'vgg16_sigmoid.pth')