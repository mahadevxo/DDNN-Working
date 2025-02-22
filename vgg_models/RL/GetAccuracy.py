import time
import torch
import random
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from ComprehensiveVGGPruner import ComprehensiveVGGPruner
class GetAccuracy:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        # print('Device: ', self.device)
    
    def replace_layers(self, features, layer_idx, replace_indices, new_layers):
        new_features = []
        for i, layer in enumerate(features):
            if i in replace_indices:
                new_features.append(new_layers[replace_indices.index(i)])
            else:
                new_features.append(layer)
        return torch.nn.Sequential(*new_features)
    
    def get_random_images(self, n=500):
        transform = transforms.Compose([
        transforms.Resize((255,255)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])
        data_folder = 'imagenet-mini/val'
        data_dataset = datasets.ImageFolder(data_folder, transform=transform)
        num_samples = n
        indices = random.sample(range(len(data_dataset)), num_samples)
        subset_dataset = Subset(data_dataset, indices)
        return DataLoader(subset_dataset, batch_size=32, shuffle=True)
    
    def prune_model(self, model_org, pruning_amount):
        model = model_org
        pruner = ComprehensiveVGGPruner(model, pruning_amount)
        return pruner.prune_all_layers()
    
    def fine_tuning(self, model):
        model = model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(3):
            running_loss = 0.0
            for data in self.get_random_images(n=300):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # print('Loss: ', running_loss)
        return model
        
    def get_accuracy(self, model_sel, sparsity):
        
        model = self.get_model(model_sel)
        model = self.prune_model(model, sparsity)
        model = self.fine_tuning(model)
        
        model.eval()
        model = model.to(self.device)
        correct = total = computation_time = 0
        
        with torch.no_grad():
            for data in self.get_random_images(n=1000):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                start_time = time.time()
                outputs = model(images)
                computation_time += time.time() - start_time
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        model_size = sum(param.numel() * param.element_size() for param in model.parameters())
        accuracy = (correct / total) * 100
        return accuracy, model_size, computation_time
    
    def get_model(self, name='vgg16'):
        model_selection = {
            'vgg16': models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1),
            'vgg11': models.vgg11(weights = models.VGG11_Weights.IMAGENET1K_V1),
            'vgg19': models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1),
            'alexnet': models.alexnet(weights = models.AlexNet_Weights.IMAGENET1K_V1),
        }
        return model_selection[name]