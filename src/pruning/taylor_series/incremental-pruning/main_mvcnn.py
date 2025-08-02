import numpy as np
import sys
sys.path.append('../../../MVCNN/')
from models import MVCNN
import torch
from tqdm import tqdm
import logging
import os
import time

device: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
# Configure logging for cleaner output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def get_exp_curve(total_sum: float, do_it: bool) -> list[float]:
    # return [total_sum]
    
    if not do_it:
        return [total_sum]
    
    if total_sum == 0:
        return [0.0] * 10
    
    # Limit total_sum to a reasonable range to avoid excessive pruning in one step
    total_sum = min(total_sum, 0.99)  # Never prune more than 99% in a single curve
    
    x = np.arange(7)
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
    # max_step_prune = 0.15
    # final_curve: list[float] = np.array([min(v, max_step_prune) for v in final_curve]).tolist() # type: ignore
    
    return final_curve

def fine_tune_model(model: torch.nn.Module, curve_value: float, org_num_filters: float, only_val: bool) -> tuple[torch.nn.Module, float, float, float]:
    if curve_value < 0:
        return model, 0, 0, 0

    from PFT_MVCNN import PruningFineTuner as pft
    pruner = pft(model, quiet=False)

    if only_val:
        return model, pruner.validate_model(), 0.0, 0.0

    if curve_value == 0.0:
        accuracy = pruner.validate_model()
        model_size = pruner.get_model_size(pruner.model)
        comp_time = pruner.get_comp_time(pruner.model)
        return pruner.model, accuracy, model_size, comp_time

    initial_filter_counts = [
        (i, layer.out_channels)
        for i, layer in enumerate(pruner.model.net_1)
        if isinstance(layer, torch.nn.modules.conv.Conv2d)
    ]
    logger.info(f"Initial filter counts: {initial_filter_counts}")

    # Prune the model
    logger.info(f"Pruning model with ratio {curve_value:.3f}")
    prev_filter_count = pruner.total_num_filters()
    logger.info(f"Requesting to prune {int(curve_value * org_num_filters)} filters from {prev_filter_count} total")

    output = pruner.prune(pruning_amount=curve_value, num_filters_to_prune=org_num_filters)
    if output is False or output is None:
        logger.info(f"Pruning failed at ratio {curve_value:.3f}")
        model_size = pruner.get_model_size(pruner.model)
        comp_time = pruner.get_comp_time(pruner.model)
        accuracy = pruner.validate_model()
        return pruner.model, accuracy, model_size, comp_time

    current_filter_count = pruner.total_num_filters()

    final_filter_counts = [
        (i, layer.out_channels)
        for i, layer in enumerate(pruner.model.net_1)
        if isinstance(layer, torch.nn.modules.conv.Conv2d)
    ]
    logger.info(f"Final filter counts: {final_filter_counts}")

    # Check if any filters were pruned
    if current_filter_count == prev_filter_count and curve_value > 0:
        logger.warning(f"No filters were actually pruned! Expected to prune {int(curve_value * org_num_filters)} filters")
        model_size = pruner.get_model_size(pruner.model)
        comp_time = pruner.get_comp_time(pruner.model)
        accuracy = pruner.validate_model()
        return pruner.model, accuracy, model_size, comp_time

    logger.info(f"Successfully pruned {prev_filter_count - current_filter_count} filters, {current_filter_count} remaining")
    logger.info(f"Accuracy of pruned model before fine-tuning: {pruner.validate_model():.2f}%")
    
    # Give ALL pruned models a small chance to recover - but very limited
    if curve_value > 0.0:
        logger.info(f"Giving pruned model ({curve_value:.1%}) a small chance to study...")
        
        # VERY minimal fine-tuning - just a tiny chance
        if curve_value > 0.8:
            epochs = 1  # Heavily pruned models get almost nothing
            base_lr = 0.000005  # Extremely low LR
            max_lr = 0.00005
        elif curve_value > 0.5:
            epochs = 1  # Moderately pruned models get a tiny bit more
            base_lr = 0.00001  # Very low LR  
            max_lr = 0.0001
        else:
            epochs = 2  # Lightly pruned models get slightly more chance
            base_lr = 0.00005  # Still very low LR
            max_lr = 0.0005
        
        optimizer = torch.optim.SGD(pruner.model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
        
        # Minimal scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr,
            steps_per_epoch=1, epochs=epochs,
            pct_start=0.3
        )

        with tqdm(range(epochs), desc="Tiny Study Time", ncols=100, colour="yellow") as pbar:
            logger.info(f"Giving {epochs} epoch(s) at LR {base_lr} (max {max_lr}) to {sum(layer.out_channels for layer in pruner.model.net_1 if isinstance(layer, torch.nn.modules.conv.Conv2d))} filters")  # Fixed: conv, not conv2d
            for epoch in pbar:
                # Check model health before training
                sample_input = torch.randn(1, 3, 224, 224).to(device)
                with torch.no_grad():
                    try:
                        sample_output = pruner.model(sample_input)
                        if torch.isnan(sample_output).any():
                            logger.error("Model produces NaN outputs, stopping training")
                            break
                    except Exception as e:
                        logger.error(f"Model forward pass failed: {e}")
                        break
                
                # Pass the optimizer to train_epoch
                pruner.train_epoch(optimizer=optimizer)

                # Step the scheduler
                scheduler.step()

                accuracy = pruner.validate_model()
                
                # Check for training collapse
                if accuracy < 0.01:  # Less than 1% accuracy suggests collapse
                    logger.warning(f"Training collapse detected at epoch {epoch+1}, accuracy: {accuracy:.4f}")
                    break
                
                pbar.set_postfix({"tiny_acc": f"{accuracy:.2f}%"})

    # Final evaluation
    accuracy = pruner.validate_model()
    model_size = pruner.get_model_size(pruner.model)
    comp_time = pruner.get_comp_time(pruner.model)

    logger.info(f"\nResults after tiny study: Acc={accuracy:.2f}%, Size={model_size:.2f}MB, Time={comp_time:.3f}s")
    x = (pruner.model, accuracy, model_size, comp_time)
    pruner.reset()
    del pruner
    return x


def get_model() -> torch.nn.Module:
    model = MVCNN.SVCNN(
        name="svcnn",
        nclasses=33,
        cnn_name="vgg11"
    )
    weights = torch.load("../../../MVCNN/MVCNN/model-mvcnn-00050.pth", map_location=device)
    model.load_state_dict(weights, strict=False)
    model = model.to(device)
    return model

def main() -> None:
    
    save_models = True
    pruning_amounts = np.array([
        0.1, 0.3, 0.5, 0.7
    ])
    # pruning_amounts = list(np.random.permutation(pruning_amounts))
    
    for part_my_part_prune in [False]:
        current_time_str = time.strftime("%Y%m%d-%H%M%S")
        logger.info(f"Experiment started at {current_time_str}")
        os.makedirs("results", exist_ok=True)
        result_path = f"results/FINAL-pruning_results_mvcnn-{current_time_str}-partbypart-{str(part_my_part_prune)}.csv"
        
        print(f"Pruning amounts to be tested: {pruning_amounts}")
        total_num_filters = sum(layer.out_channels for layer in get_model().net_1 if isinstance(layer, torch.nn.modules.conv.Conv2d))  # type: ignore
        logger.info(f"Starting pruning experiment with {len(pruning_amounts)} pruning ratios")
        # np.random.shuffle(pruning_amounts)
        # print(f"Pruning amounts to be tested: {pruning_amounts}")
        # Initialize results file
        with open(result_path, 'w') as f:
            f.write("Pruning_Amount,Accuracy,Model_Size_MB,Computation_Time, Number Of Filters\n")
            
        logger.info(f"Results will be saved to {result_path}")
        
        model = get_model()
        print(f"Initial Accuracy: {fine_tune_model(model, 0.0, total_num_filters, only_val=True)[1]:.4f}")
        del model
        
        # Main pruning loop
        with tqdm(pruning_amounts, desc="Pruning Ratios", ncols=150) as pbar_outer:
            for pruning_amount in pbar_outer:
                try:
                    model = get_model()
                    pbar_outer.set_postfix({"ratio": f"{pruning_amount:.2f}"})
                    logger.info(f"\n{'='*50}\nProcessing pruning ratio: {pruning_amount:.2f}\nTrainable Filters: {sum(layer.out_channels for layer in model.net_1 if isinstance(layer, torch.nn.modules.conv.Conv2d))}\n{'='*50}\n")  # type: ignore # Fixed: conv, not conv2d
                    
                    if pruning_amount == 0.0:
                        # Baseline (unpruned) evaluation
                        model, final_acc, model_size, comp_time = fine_tune_model(model=model, curve_value=0.0, org_num_filters=total_num_filters, only_val=False)
                        num_filters_present = sum(layer.out_channels for layer in model.net_1 if isinstance(layer, torch.nn.modules.Conv2d))  # type: ignore # Fixed: conv, not conv2d
                        print(f"Baseline model size: {model_size:.4f} MB, Accuracy: {final_acc:.4f}, Computation Time: {comp_time:.4f} seconds")
                        with open(result_path, 'a') as f:
                            f.write(f"{pruning_amount:.2f},{final_acc:.4f},{model_size:.4f},{comp_time:.4f},{num_filters_present}\n")
                        continue
                    
                    # Get exponential curve values for progressive pruning
                    curve = get_exp_curve(pruning_amount, do_it=part_my_part_prune)
                    curve = [c for c in curve if c > 0]  # Filter out zeros
                    
                    if not curve:
                        continue
                    
                    # Process each step of the curve
                    final_metrics = None
                    for i, curve_value in enumerate(curve):
                        logger.info(f"\nStep {i+1}/{len(curve)}: Pruning ratio = {curve_value:.3f}\nTrainable Filters: {sum(layer.out_channels for layer in model.net_1 if isinstance(layer, torch.nn.modules.conv.Conv2d))}\n")  # type: ignore # Fixed: conv, not conv2d
                        model, accuracy, model_size, comp_time = fine_tune_model(model=model, curve_value=curve_value, org_num_filters=total_num_filters, only_val=False)
                        num_filters_present = sum(layer.out_channels for layer in model.net_1 if isinstance(layer, torch.nn.modules.Conv2d))  # type: ignore # Fixed: conv, not conv2d
                        final_metrics = (pruning_amount, accuracy, model_size, comp_time, num_filters_present)
                    
                    # Save results
                    if final_metrics:
                        with open(result_path, 'a') as f:
                            f.write(f"{final_metrics[0]:.2f},{final_metrics[1]:.4f},{final_metrics[2]:.4f},"
                                    f"{final_metrics[3]:.4f}, {final_metrics[4]}\n")
                    print(model)
                    if save_models:
                        model_save_path = f"models/mvcnn_pruned_{pruning_amount:.2f}.pth"
                        torch.save(model, model_save_path)
                        logger.info(f"Model saved to {model_save_path}")
                    del model  # Clear model from memory
                    del final_metrics  # Clear metrics from memory
                        
                except Exception as e:
                    logger.error(f"Error at pruning_amount={pruning_amount}: {str(e)}")
                    with open(result_path, 'a') as f:
                        f.write(f"{pruning_amount:.2f},Error,0,0,0,0\n")
        
        logger.info(f"\nExperiment completed. Results saved to {result_path}")

if __name__ == "__main__":
    main()