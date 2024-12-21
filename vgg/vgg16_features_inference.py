import torch
from torchvision.models import vgg16
import csv
import time

class VGG16_Inference():
    def __init__(self):
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.model = vgg16(pretrained=True).to(self.device)
        self.model.eval()
    
    def create_images(self, count):
        images = []
        for _ in range(0, count):
            images.append(torch.rand(3, 224, 224))
        return images
    
    def extract_features(self, model):
        return model.features.to(self.device)
    
    def forward(self, model, image):
        image = image.to(self.device)
        with torch.no_grad():
            return model(image)
    
    def run(self):
        input_images = 100
        print(f'Running inference on {input_images} images')
        images = self.create_images(input_images)
        model = self.extract_features(self.model)
        
        print(f'Using {self.device}')
        
        with open(f'vgg16_{self.device}.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['image', 'time'])
            print("Inference Started")
            for i, image in enumerate(images):
                start = time.time()
                image = image.unsqueeze(0)
                self.forward(model, image)  
                end = time.time()
                writer.writerow([i, end - start])
        print('Done')

if __name__ == '__main__':
    vgg16_inference = VGG16_Inference()
    vgg16_inference.run()
