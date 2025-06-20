import numpy as np
import torchvision
from torchvision.models import vgg16
import torch
import types
from tqdm import tqdm
import logging
import os
import sys
import time

# Configure logging for cleaner output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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
    
    # Limit total_sum to a reasonable range to avoid excessive pruning in one step
    total_sum = min(total_sum, 0.99)  # Never prune more than 99% in a single curve
    
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
    
    # Ensure no step has more than 15% pruning
    max_step_prune = 0.15
    final_curve: list[float] = np.array([min(v, max_step_prune) for v in final_curve]).tolist() # type: ignore
    
    return final_curve

def fine_tune_model(model, curve_value) -> tuple[float, float, float]:
    if curve_value < 0:
        return 0, 0, 0

    from PFT import PruningFineTuner as pft
    pruner = pft(model, quiet=False)
    
    if curve_value == 0.0:
        accuracy = pruner.get_val_accuracy(model)
        model_size = pruner.get_model_size(model)
        comp_time = pruner.get_comp_time(model)
        return accuracy, model_size, comp_time
    
    # Prune the model
    logger.info(f"Pruning model with ratio {curve_value:.3f}")
    prev_filter_count = pruner.total_num_filters()
    model = pruner.prune(pruning_amount=curve_value)
    current_filter_count = pruner.total_num_filters()
    
    # Check if any filters were pruned
    if current_filter_count == prev_filter_count and curve_value > 0:
        logger.info(f"No more filters can be pruned at ratio {curve_value:.3f}")
        model_size = pruner.get_model_size(model)
        comp_time = pruner.get_comp_time(model)
        return 0.0, model_size, comp_time
    
    # Fine-tune the pruned model
    logger.info(f"Fine-tuning pruned model ({curve_value:.3f})")
    
    accuracy_previous = []
    epochs = 15
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.005, 
        steps_per_epoch=1, epochs=epochs,
        pct_start=0.3  # Warmup for 30% of training
    )
    
    with tqdm(range(epochs), desc="Training", ncols=100, colour="green") as pbar:
        for epoch in pbar:
            # Pass the optimizer to train_epoch
            model = pruner.train_epoch(optimizer=optimizer)
            
            # Step the scheduler
            scheduler.step()
            
            accuracy = pruner.get_val_accuracy(model=model)
            
            accuracy_previous.append(accuracy)
            if len(accuracy_previous) > 4:
                accuracy_previous.pop(0)
                
            pbar.set_postfix({"val acc": f"{accuracy:.2f}%"})
            
            # Early stopping check (convergence)
            if len(accuracy_previous) >= 3 and accuracy-1.0 <= np.mean(accuracy_previous) <= accuracy+1.0:
                logger.info(f"Converged at epoch {epoch+1}/{epochs}")
                break
                
    # Final evaluation
    accuracy = pruner.get_val_accuracy(model=model)
    model_size = pruner.get_model_size(model=model)
    comp_time = pruner.get_comp_time(model=model)
    
    logger.info(f"Results: Acc={accuracy:.2f}%, Size={model_size:.2f}MB, Time={comp_time:.3f}s")
    return accuracy, model_size, comp_time


def get_model() -> torch.nn.Module:
    model = vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    model.net_1 = model.features
    model.net_2 = model.classifier
    del model.classifier
    del model.features
    model.forward = types.MethodType(forward_override, model)
    return model

def main() -> None:
    # Create results directory
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    logger.info(f"Experiment started at {current_time_str}")
    os.makedirs("results", exist_ok=True)
    result_path = f"results/pruning_results-{current_time_str}.csv"
    
    # Define pruning range
    pruning_amounts = np.arange(0, 1, 0.05)
    
    done_amounts = {0.30, 0.25, 0.60, 0.45, 0.20, 0.15, 0.50,}
    
    pruning_amounts = [p for p in pruning_amounts if p not in done_amounts]
    
    print(f"Pruning amounts to be tested: {pruning_amounts}")
    
    logger.info(f"Starting pruning experiment with {len(pruning_amounts)} pruning ratios")
    np.random.shuffle(pruning_amounts)
    
    # Initialize results file
    with open(result_path, 'w') as f:
        f.write("Pruning_Amount,Accuracy,Model_Size_MB,Computation_Time\n")
    
    # Main pruning loop
    with tqdm(pruning_amounts, desc="Pruning Ratios", ncols=150) as pbar_outer:
        for pruning_amount in pbar_outer:
            try:
                model = get_model()
                pbar_outer.set_postfix({"ratio": f"{pruning_amount:.2f}"})
                logger.info(f"\n{'='*50}\nProcessing pruning ratio: {pruning_amount:.2f}\nTrainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n{'='*50}\n")
                
                if pruning_amount == 0.0:
                    # Baseline (unpruned) evaluation
                    final_acc, model_size, comp_time = fine_tune_model(model, 0.0)
                    print(f"Baseline model size: {model_size:.4f} MB, Accuracy: {final_acc:.4f}, Computation Time: {comp_time:.4f} seconds")
                    with open(result_path, 'a') as f:
                        f.write(f"{pruning_amount:.2f},{final_acc:.4f},{model_size:.4f},{comp_time:.4f}\n")
                    continue
                
                # Get exponential curve values for progressive pruning
                curve = get_exp_curve(pruning_amount)
                curve = [c for c in curve if c > 0]  # Filter out zeros
                
                if not curve:
                    continue
                
                # Process each step of the curve
                final_metrics = None
                for i, curve_value in enumerate(curve):
                    logger.info(f"\nStep {i+1}/{len(curve)}: Pruning ratio = {curve_value:.3f}\nTrainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
                    accuracy, model_size, comp_time = fine_tune_model(model, curve_value)
                    final_metrics = (pruning_amount, accuracy, model_size, comp_time)
                
                # Save results
                if final_metrics:
                    with open(result_path, 'a') as f:
                        f.write(f"{final_metrics[0]:.2f},{final_metrics[1]:.4f},{final_metrics[2]:.4f},"
                                f"{final_metrics[3]:.4f}\n")
                    
            except Exception as e:
                logger.error(f"Error at pruning_amount={pruning_amount}: {str(e)}")
                with open(result_path, 'a') as f:
                    f.write(f"{pruning_amount:.2f},Error,0,0,0\n")
    
    logger.info(f"\nExperiment completed. Results saved to {result_path}")

if __name__ == "__main__":
    main()