from torchvision import models as models
from PruningFineTuner import PruningFineTuner
import numpy
import torch
import gc

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    pruning_amounts = numpy.arange(0.0, 60.0, 2.0)
    print(f"Number of pruning percentages: {len(pruning_amounts)}")
    
    with open("pruning_vs_comp_time.csv", mode='w') as file:
        file.write("Pruning Amount, Pre-Accuracy, Accuracy, Pre-Pruning Time, Post-Pruning Time, Size\n")
        for pruning_amount in pruning_amounts:
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
            pruning_fine_tuner = PruningFineTuner(model)
            data = pruning_fine_tuner.prune(pruning_amount)
            pre = data[0]
            pre_acc, pre_comp = pre[0], pre[1]
            acc, comp_time, size = data[1], data[2], data[3]
            entry = f"{pruning_amount}, {pre_acc}, {acc}, {pre_comp}, {comp_time}, {size}"
            print(entry)
            file.write(entry + '\n')
            file.flush()
            del pruning_fine_tuner
            del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    print("Done")

if __name__ == '__main__':
    main()