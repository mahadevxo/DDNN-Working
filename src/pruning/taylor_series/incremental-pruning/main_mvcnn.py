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
    
    total_sum = min(total_sum, 1.0) 
    
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

    if curve_value > 0:
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
    if True:
        logger.info(f"Giving pruned model ({curve_value:.1%}) a small chance to study...")
        epochs = 2 
        base_lr = 0.001
        max_lr = 0.005
        
        optimizer = torch.optim.SGD(pruner.model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
        
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
    # Instantiate single‐view backbone
    base_cnn = MVCNN.SVCNN(
        name="svcnn",
        nclasses=33,
        cnn_name="vgg11"
    )
    weights = torch.load("../../../MVCNN/MVCNN/model-mvcnn-00050.pth", map_location=device)
    base_cnn.load_state_dict(weights, strict=False)
    base_cnn = base_cnn.to(device)
    
    # Wrap into MVCNN so forward() never touches base_cnn.net
    mvcnn = MVCNN.MVCNN(
        name="mvcnn",
        model=base_cnn,
        nclasses=33,
        cnn_name="vgg11",
        num_views=12
    )
    return mvcnn.to(device)

def main() -> None:
    test_type = int(input("Enter test type (save model and all - 0/ normal test - 1): "))
    save_models = True if test_type == 0 else False
    pruning_amounts = [0.1, 0.3, 0.5, 0.9, 0.0, 0.7] if save_models else np.arange(0.0, 1, 0.05).tolist()
    print(f"Save models? {save_models}")
    # noqa: E702
    for part_my_part_prune in [False]:
        current_time_str = time.strftime("%Y%m%d-%H%M%S")
        logger.info(f"Experiment started at {current_time_str}")
        os.makedirs("results", exist_ok=True)
        result_path = f"results/FINAL-pruning_results_mvcnn-{current_time_str}-partbypart-{str(part_my_part_prune)}.csv"
        
        print(f"Pruning amounts to be tested: {pruning_amounts}")
        # total filters before any pruning, excluding last conv2d
        conv_layers = [
            l for l in get_model().net_1 # type: ignore
            if isinstance(l, torch.nn.modules.conv.Conv2d)
        ]
        if conv_layers:
            counts = [l.out_channels for l in conv_layers[:-1]]
        else:
            counts = []
        total_num_filters = sum(counts)  # type: ignore
    
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
                        model, final_acc, model_size, comp_time = fine_tune_model(
                            model=model, curve_value=-1,
                            org_num_filters=total_num_filters, only_val=False
                        )
                        num_filters_present = sum(layer.out_channels for layer in model.net_1 if isinstance(layer, torch.nn.modules.Conv2d))  # type: ignore
                        print(f"Baseline model size: {model_size:.4f} MB, Accuracy: {final_acc:.4f}, Computation Time: {comp_time:.4f} seconds")
                        with open(result_path, 'a') as f:
                            f.write(f"{pruning_amount:.2f},{final_acc:.4f},{model_size:.4f},{comp_time:.4f},{num_filters_present}\n")
                        
                        model_save_path = f"models/mvcnn_pruned_{pruning_amount:.2f}.pth"
                        scripted_model = torch.jit.script(model)
                        scripted_model.save(model_save_path)
                        
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
                        num_filters_present = sum(layer.out_channels for layer in model.net_1 if isinstance(layer, torch.nn.modules.Conv2d))  # type: ignore
                        final_metrics = (pruning_amount, accuracy, model_size, comp_time, num_filters_present)
                    
                    # Save results
                    if final_metrics:
                        with open(result_path, 'a') as f:
                            f.write(f"{final_metrics[0]:.2f},{final_metrics[1]:.4f},{final_metrics[2]:.4f},"
                                    f"{final_metrics[3]:.4f}, {final_metrics[4]}\n")
                    print(model)
                    if save_models:
                        model_save_path = f"models/mvcnn_pruned_{pruning_amount:.2f}.pth"
                        # Ensure TorchScript module is saved via .save(), not torch.save
                        scripted_model = torch.jit.script(model)
                        scripted_model.save(model_save_path)
                        
                    del model  # Clear model from memory
                    del final_metrics  # Clear metrics from memory
                        
                except Exception as e:
                    logger.error(f"Error at pruning_amount={pruning_amount}: {str(e)}")
                    with open(result_path, 'a') as f:
                        f.write(f"{pruning_amount:.2f},Error,0,0,0,0\n")
        
        logger.info(f"\nExperiment completed. Results saved to {result_path}")

if __name__ == "__main__":
    main()