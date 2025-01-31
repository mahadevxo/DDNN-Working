import functions_prune
import numpy
import shutil
import torch
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_device():
    device_available = 'cuda' if torch.cuda.is_available() else 'CoreML' if torch.backends.mps.is_available else 'CPU'

    return bool(
        int(input(f'\nEnter 0 to use PyTorch on CPU or 1 to use {device_available}: '))
    )

def main():
    functionsprune = functions_prune.FunctionsPrune()
    if os.path.exists('coremlmodel.mlpackage/'):
        shutil.rmtree('coremlmodel.mlpackage/')
    sparsities = numpy.arange(0, 1.05, 0.05)
    models = ['MobileNetV3 Large', 'AlexNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'ConvNeXt Large']
    coreml = get_device()
    for model_using in models:
        params = None
        computation_times = []
        print(f'{bcolors.BOLD}{bcolors.FAIL}Model Used: {model_using}'.center(shutil.get_terminal_size().columns))
        for sparsity in sparsities:
            model = functionsprune.get_model(model_using)
            model.eval()
            model, p = functionsprune.prune_model(model, sparsity)
            params = p if params is None else params
            if coreml:
                model = functionsprune.convert_to_coreml(model)
                shutil.rmtree('coremlmodel.mlpackage/')
            inference_time = functionsprune.computation_time(model, coreml, 100)
            print(
                f'{bcolors.WARNING}Model: {model_using}, Params: {params}, Sparsity: {sparsity:.2f}, {bcolors.OKGREEN}Inference time: {inference_time:.2f} ms'
            )
            computation_times.append(inference_time)

        functionsprune.plot_graph(computation_times, sparsities, model_using, params)
        params = None

        with open(f'computation_times_{model_using}.csv', 'w') as f:
            f.write('Sparsity,Inference Time\n')
            for sparsity, inference_time in zip(sparsities, computation_times):
                f.write(f'{sparsity},{inference_time}\n')


if __name__ == '__main__':
    main()