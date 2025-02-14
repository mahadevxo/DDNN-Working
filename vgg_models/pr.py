import torch.nn as nn
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

model = models.vgg16(pretrained=False)
model.classifier[len(model.classifier)-1] = nn.Linear(4096, 2)

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    model = model.to('cuda')
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    model = model.to('mps')
    print("Using MPS")
else:
    device = torch.device('cpu')
    model = model.to('cpu')
    print("Using CPU")


train_folder = 'cat-dogs-dataset/training_set/training_set/'
test_folder = 'cat-dogs-dataset/test_set/test_set/'

transform = transforms.Compose([
    transforms.Resize((250,250)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

dataset = ImageFolder(train_folder, transform=transform)

eval_dataset = ImageFolder(test_folder, transform=transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True, num_workers=4)

train_size = int(0.9 * len(dataset))  
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
eval_loader = DataLoader(eval_dataset, batch_size=200, shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
epochs = 10

accuracy_list, loss_list = [], []

for epoch in range(epochs):
    model.train() 
    running_loss = 0.0
    correct, total = 0, 0
    print(f"Epoch {epoch+1}/{epochs}")
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = (correct / total) * 100
    epoch_loss = running_loss / len(train_loader)
    accuracy_list.append(accuracy)
    loss_list.append(epoch_loss)
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
