import numpy as np
import torchvision
from torchvision.models import vgg16
import torch
import types
from tqdm import tqdm

def forward_override(self, x):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected input to be a torch.Tensor, got {type(x)} instead.")
    x = self.net_1(x)
    x = x.view(x.size(0), -1)
    x = self.net_2(x)
    return x

def get_exp_curve(total_sum) -> list[float]:
    if total_sum == 0:
        return [0.0] * 10
    
    x = np.arange(10)
    decay_target_ratio = 0.01
    
    k_rate = -np.log(decay_target_ratio) / 9
    curve_raw = np.exp(-k_rate * x)
    shift_amount = curve_raw[-1]    
    curve_shifted = curve_raw - shift_amount
    sum_of_shifted = np.sum(curve_shifted)
    scaling_factor = total_sum / sum_of_shifted
    final_curve = curve_shifted * scaling_factor
    final_curve[-1] = 0.0
    return final_curve.tolist()

def fine_tune_model(model, curve_value):
    if curve_value < 0:
        raise ValueError(f"Curve value {curve_value} is negative, which is not allowed.")

    from PFT import PruningFineTuner as pft
    pruner = pft(model)
    
    if curve_value == 0.0:
        print("No Pruning Required")
        accuracy = pruner.get_val_accuracy(model=model)
        model_size = pruner.get_model_size(model=model)
        comp_time = pruner.get_comp_time(model=model)
        return accuracy, model_size, comp_time
    
    print(f"Pruning model: {curve_value:.2f}")

    model = pruner.prune(
        pruning_amount=curve_value,
        only_model=False,
        prune_targets=None
    )

    print(f"Fine-tuning at {curve_value:.2f}...")

    accuracy_previous = []
    pbar = tqdm(range(15), desc="Training epochs")
    
    for epoch in pbar:
        try:
            model = pruner.train_epoch()
            accuracy = pruner.get_val_accuracy(model=model)
            model_size = pruner.get_model_size(model=model)
            comp_time = pruner.get_comp_time(model=model)

            accuracy_previous.append(accuracy)
            if len(accuracy_previous) > 4:
                accuracy_previous.pop(0)

            pbar.set_postfix({"accuracy": f"{accuracy:.2f}%"})

            if accuracy-1e-2 <=np.mean(accuracy_previous) <= accuracy+1e-2:
                break

        except Exception as e:
            print(f"Training error: {e}")
            break
    
    accuracy = pruner.get_val_accuracy(model=model)
    model_size = pruner.get_model_size(model=model)
    comp_time = pruner.get_comp_time(model=model)

    return accuracy, model_size, comp_time

def main() -> None:
    pruning_amounts = np.arange(0, 1, 0.05)
    print(f"Testing {len(pruning_amounts)} pruning amounts")
    
    with open('results.csv', 'w') as f:
        f.write("Pruning Amount, Curve Value, Final Accuracy, Model Size (MB), Computation Time (seconds)\n")
    
    model = vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    
    model.net_1 = model.features
    model.net_2 = model.classifier
    del model.classifier
    del model.features
    model.forward = types.MethodType(forward_override, model)
    
    pbar_outer = tqdm(pruning_amounts, desc="Pruning amounts")
    for pruning_amount in pbar_outer:
        try:
            pbar_outer.set_postfix({"amount": f"{pruning_amount:.2f}"})
            
            if pruning_amount == 0.0:
                final_acc, model_size, comp_time = fine_tune_model(model, 0.0)
                with open('results.csv', 'a') as f:
                    f.write(f"{pruning_amount}, 0.0, {final_acc}, {model_size}, {comp_time}\n")
                continue
            
            curve = get_exp_curve(pruning_amount)
            
            if not curve:
                continue
                
            final_acc, model_size, comp_time = 0, 0, 0
            
            pbar_inner = tqdm(curve, desc=f"Curves for {pruning_amount:.2f}")
            for curve_value in pbar_inner:
                if curve_value == 0:
                    continue
                
                pbar_inner.set_postfix({"curve": f"{curve_value:.3f}"})
                accuracy, model_size, comp_time = fine_tune_model(model, curve_value)
                final_acc = accuracy
                
            with open('results.csv', 'a') as f:
                f.write(f"{pruning_amount}, {curve_value}, {final_acc}, {model_size}, {comp_time}\n")
                
        except Exception as e:
            print(f"Error at pruning_amount={pruning_amount}: {e}")
            with open('results.csv', 'a') as f:
                f.write(f"{pruning_amount}, Error: {str(e)}\n")
            continue

if __name__ == "__main__":
    main()