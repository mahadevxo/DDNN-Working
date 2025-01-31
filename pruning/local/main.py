import functions_prune
import numpy
import shutil
import torch

#     input_shape (tuple): The shape of the input tensor.

functionsprune = functions_prune.FunctionsPrune()
shutil.rmtree('coremlmodel.mlpackage/')

sparsities = numpy.arange(0, 1.05, 0.05)
computation_times = []

device_available = 'cuda' if torch.cuda.is_available() else 'CoreML' if torch.backends.mps.is_available else 'CPU'

use_core_ml = bool(int(input(f'Enter 1 to use {device_available}, 0 to use PyTorch: ')))

models = ['vgg11', 'vgg16', 'alexnet', 'mobilenetv3']
for model_name in models:
    for sparsity in sparsities:
        model = functionsprune.get_model(model_name)
        model = functionsprune.prune_model(model, sparsity)
        if use_core_ml:
            model = functionsprune.convert_to_coreml(model)
            shutil.rmtree('coremlmodel.mlpackage/')
        inference_time = functionsprune.computation_time(model, use_core_ml)
        print(f'Sparsity: {sparsity}, Inference time: {inference_time:.2f} ms')
        computation_times.append(inference_time)

    functionsprune.plot_graph(computation_times, sparsities, model_name)

    with open(f'computation_times_{model_name}.csv', 'w') as f:
        f.write('Sparsity,Inference Time\n')
        for sparsity, inference_time in zip(sparsities, computation_times):
            f.write(f'{sparsity},{inference_time}\n')