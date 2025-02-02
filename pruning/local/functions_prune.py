import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import coremltools
import time
import matplotlib.pyplot as plt
import numpy
import PIL

class FunctionsPrune:
    
    def __init__(self):
        pass
        
    def get_model(self, model_name='vgg11'):
        # print('Model:', model_name)
        if model_name == 'MobileNetV3 Large':
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        elif model_name == 'AlexNet':
            model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        elif model_name == 'VGG11':
            model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        elif model_name == 'VGG13':
            model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
        elif model_name == 'VGG16':
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        elif model_name == 'VGG19':
            model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        elif model_name == 'ConvNeXt Large':
            model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
        else:
            raise ValueError(f'Unsupported model: {model_name}')

        return model

    def prune_model(self,model, prune_rate):
        for _, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=prune_rate)
        names_params = list(model.named_parameters())
        return model, len(names_params)

    def convert_to_coreml(self, model, input_shape=(1, 3, 224, 224)):
        model.eval()
        traced_model = torch.jit.trace(model, torch.randn(input_shape))
        coremlmodel= coremltools.convert(
            traced_model,
            source='pytorch',
            convert_to='mlprogram',
            minimum_deployment_target = coremltools.target.iOS18,
            inputs=[coremltools.ImageType(name='image', shape=input_shape)],
        )
        coremlmodel.save("coremlmodel.mlpackage")
        return coremltools.models.MLModel('coremlmodel.mlpackage')
        
    def computation_time(self, model, use_core_ml, count):
        images = []
        if use_core_ml:
            for _ in range(count):
                image = torch.randn(3, 224, 224).numpy().astype(numpy.float32)
                image = image.transpose(1, 2, 0)
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(numpy.uint8)
                image = PIL.Image.fromarray(image)
                images.append({"image": image})
        else:
            model.eval()
            images = torch.randn(100, 3, 224, 224)

        start_time = time.time()
        
        if use_core_ml:
            for image in images:
                model.predict(image)
        else:
            with torch.no_grad():
                _ = model(images)
        end_time = time.time()
        return end_time - start_time

    def plot_graph(self, computation_time, sparsities, model_name, params):
        plt.plot(sparsities, computation_time)
        plt.xlabel('Sparsity')
        plt.ylabel('Inference Time')
        plt.title(f'\n{model_name}->{params}\nInference Time vs. Sparsity')
        plt.grid()
        plt.savefig(f'inference_time_vs_sparsity_{model_name}.png')
        plt.clf()
        # plt.show()