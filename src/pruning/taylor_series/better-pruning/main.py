import numpy as np
from models.MVCNN import SVCNN
import torch

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
    
    print("Pruning model with pruning amount:", curve_value)

    model = pruner.prune(
        pruning_amount=curve_value,
        only_model=False,
        prune_targets=None
    )

    print("Pruned model successfully with pruning amount:", curve_value, "Starting Fine Tuning")

    accuracy_previous = []

    while True:
        try:
            model = pruner.train_epoch()
            accuracy = pruner.get_val_accuracy(model=model)
            model_size = pruner.get_model_size(model=model)
            comp_time = pruner.get_comp_time(model=model)

            accuracy_previous.append(accuracy)
            if len(accuracy_previous) > 4:
                accuracy_previous.pop(0)

            if accuracy-1e-2 <=np.mean(accuracy_previous) <= accuracy+1e-2:
                print(f"Model converged with accuracy: {accuracy}, model size: {model_size} MB, computation time: {comp_time} seconds")
                break

            if len(accuracy_previous) > 10:
                print(f"Stopping after {len(accuracy_previous)} steps")
                break
        except Exception as e:
            print(f"An error occurred during training: {e}")
            break
    
    print("Fine tuning completed successfully")

    accuracy = pruner.get_val_accuracy(model=model)
    model_size = pruner.get_model_size(model=model)
    comp_time = pruner.get_comp_time(model=model)

    return accuracy, model_size, comp_time

def main() -> None:
    pruning_amounts = np.arange(0, 1, 0.05)
    print(f"Testing {len(pruning_amounts)} pruning amounts")
    print(f"Pruning amounts: {pruning_amounts}")
    
    with open('results.csv', 'w') as f:
        f.write("Pruning Amount, Curve Value, Final Accuracy, Model Size (MB), Computation Time (seconds)\n")
    
    model = SVCNN(name='SVCNN')
    weights = torch.load('model.pth', map_location='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.load_state_dict(weights)
    
    for pruning_amount in pruning_amounts:
        try:
            if pruning_amount == 0.0:
                final_acc, model_size, comp_time = fine_tune_model(model, 0.0)
                with open('results.csv', 'a') as f:
                    f.write(f"{pruning_amount}, {final_acc}, {model_size}, {comp_time}\n")
                continue
            
            curve = get_exp_curve(pruning_amount)
            
            print(f"Pruning amount: {pruning_amount}, Curve: {curve}")
            if not curve:
                print(f"Skipping pruning amount {pruning_amount} as the curve is empty.")
                continue
            final_acc, model_size, comp_time = 0, 0, 0
            for curve_value in curve:
                if curve_value == 0:
                    print(f"Skipping curve value {curve_value} as it is zero.")
                    continue
                
                accuracy, model_size, comp_time = fine_tune_model(model, curve_value)
                print(f"Final accuracy: {accuracy}, model size: {model_size} MB, computation time: {comp_time} seconds for pruning amount {pruning_amount} and curve value {curve_value}")
                final_acc = accuracy
                model_size = model_size
                comp_time = comp_time
                
            with open('results.csv', 'a') as f:
                f.write(f"{pruning_amount}, {final_acc}, {model_size}, {comp_time}\n")
        except Exception as e:
            print(f"An error occurred while processing pruning amount {pruning_amount}: {e}")
            with open('results.csv', 'a') as f:
                f.write(f"{pruning_amount}, Error: {str(e)}\n")
            continue
    # Final message after processing all pruning amounts
    print("All pruning amounts processed successfully.")

if __name__ == "__main__":
    main()
    print("Main function executed successfully.")