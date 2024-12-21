import torch
from torchvision.models import vgg16
import csv
import time

class VGG16_Inference():
    def __init__(self):
        self.model = vgg16(pretrained=True)
        self.model.eval()
    
    def create_images(self, count):
        images = []
        for _ in range(0, count):
            images.append(torch.rand(3, 224, 224))
        return images
    
    def extract_features(self, model):
        return model.features
    
    def forward(self, model, images):
        return model(images)
    
    def run(self):
        input_images = 100
        images = self.create_images(input_images)
        model = self.extract_features(self.model)
        with open('vgg16.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['image', 'time'])
            print("Inference Started")
            for i, image in enumerate(images):
                start = time.time()
                self.forward(model, image)
                end = time.time()
                writer.writerow([i, end - start])
        print('Done')

if __name__ == '__main__':
    vgg16_inference = VGG16_Inference()
    vgg16_inference.run()