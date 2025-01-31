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
    
    def __init__(self, input_size = (1, 3, 224, 224)):
        self.input_size = input_size
        self.model = self.get_model()
        self.model = self.prune_model(self.model, 0.5)
        self.model = self.convert_to_coreml(self.model, self.input_size)

        
    def get_model(self, model_name='vgg11'):
        """Retrieves a pretrained model from the torchvision.models library.

        This function returns a specified pretrained model from torchvision.models.
        It supports 'vgg11', 'vgg16', 'alexnet', and 'mobilenetv3'.

        Args:
            model_name (str): The name of the pretrained model to retrieve.

        Returns:
            nn.Module: The pretrained model.

        Raises:
            ValueError: If the provided model_name is not supported.
        """
        print('Model:', model_name)
        
        if model_name == 'vgg11':
            return models.vgg11(pretrained=True)
        elif model_name == 'vgg16':
            return models.vgg16(pretrained=True)
        elif model_name == 'alexnet':
            return models.alexnet(pretrained=True)
        elif model_name == 'mobilenetv3':
            return models.mobilenet_v3_small(pretrained=True)
        else:
            raise ValueError('Unknown model name')

    def prune_model(self,model, prune_rate):
        """Prunes convolutional layers in a given model using L1 unstructured pruning.

        This function iterates through the model's modules and applies L1 unstructured
        pruning to the weights of convolutional layers (nn.Conv2d) based on the
        specified pruning rate.  It then prints the total number of parameters in the model.

        Args:
            model (nn.Module): The model to prune.
            prune_rate (float): The pruning rate (proportion of weights to prune).

        Returns:
            nn.Module: The pruned model.
        """
        for _, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=prune_rate)
        names_params = list(model.named_parameters())
        return model, names_params

    def convert_to_coreml(self, model, input_shape=(1, 3, 224, 224)):
        """Converts a PyTorch model to CoreML format.

        This function traces the given PyTorch model using a random input tensor and
        converts it to CoreML format using coremltools.

        Args:
            model (nn.Module): The PyTorch model to convert.
            input_shape (tuple): The shape of the input tensor.

        Returns:
            coremltools.models.MLModel: The converted CoreML model.
        """
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
        
    def computation_time(self, model, use_core_ml):
        images = []
        if use_core_ml:
            for _ in range(100):
                image = torch.randn(3, 224, 224).numpy().astype(numpy.float32)
                image = image.transpose(1, 2, 0)
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(numpy.uint8)
                image = PIL.Image.fromarray(image)
                images.append({"image": image})
        else:
            images = [torch.randn(1, 3, 224, 224) for _ in range(100)]

        start_time = time.time()
        for image_dict in images:
            if use_core_ml:
                model.predict(image_dict)
            else:
                model(image_dict)
        end_time = time.time()
        return end_time - start_time

    def plot_graph(self, computation_time, sparsities, model_name):
        """Plots a graph of inference time vs. sparsity.
        This function plots the given computation time against a range of
        sparsities and saves the plot as 'inference_time_vs_sparsity.png'.

        Args:
            computation_time (list): A list of inference times.
        """
        plt.plot(sparsities, computation_time)
        plt.xlabel('Sparsity')
        plt.ylabel('Inference Time')
        plt.title('Inference Time vs. Sparsity')
        plt.savefig(f'inference_time_vs_sparsity_{model_name}.png')
        plt.show()