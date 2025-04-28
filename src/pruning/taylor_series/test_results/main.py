import time
import numpy as np
from MVCNN.models import MVCNN
from MVCNN.tools.ImgDataset import SingleImgDataset
import gc
import torch
from copy import deepcopy
from PFT import PruningFineTuner
import os

class Testing:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        test_dataset = SingleImgDataset(
            root_dir='ModelNet40-12View/*/test',
            scale_aug=False,
            rot_aug=False,
            num_models=1000,
            num_views=12
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        
    def get_model(self) -> torch.nn.Module:
        model = MVCNN.SVCNN(
            'svcnn',
        )
        model.load_state_dict(torch.load('model-00030.pth', map_location=self.device))
        model = model.to(self.device)

        return deepcopy(model)
    
    def get_size(self, model):
        param_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        buffer_size = sum(
            buffer.nelement() * buffer.element_size() for buffer in model.buffers()
        )
        return (param_size + buffer_size) / 1024**2

    def _clear_memory(self) -> None:
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()
        gc.collect()
    

    def get_dataset(self, train_dataset=True, test_dataset=False, comp_time_dataset=False) -> torch.utils.data.DataLoader:
        # Load dataset
        if train_dataset:
            dataset = SingleImgDataset(
                root_dir='ModelNet40-12View/*/train',
                scale_aug=False,
                rot_aug=True,
                num_models=1000,
                num_views=12,
            )
        elif test_dataset or comp_time_dataset:
            dataset = SingleImgDataset(
                root_dir='ModelNet40-12View/*/test',
                scale_aug=False,
                rot_aug=False,
                num_models=1000,
                num_views=12,
            )

        total_models = len(dataset.filepaths) // 12

        # Use deterministic sampling instead of random
        subset_size = 0.1 if train_dataset else 1.0 if test_dataset else 0.03 if comp_time_dataset else None

        if subset_size is None:
            raise ValueError("Invalid subset size")

        subset_size = int(total_models * subset_size)

        # Use stratified sampling to ensure class representation
        if train_dataset:
            # Group filepaths by class
            class_filepaths = {}
            for filepath in dataset.filepaths:
                class_name = filepath.split('/')[1]
                if class_name not in class_filepaths:
                    class_filepaths[class_name] = []
                class_filepaths[class_name].append(filepath)

            # Ensure we have all 33 classes
            if len(class_filepaths) < 33:
                print(f"Warning: Only {len(class_filepaths)} classes found in dataset")
                return False

            # Take equal samples from each class
            models_per_class = max(1, subset_size // len(class_filepaths))
            new_filepaths = []

            for filepaths in class_filepaths.values():
                # Group by model (every 12 views is one model)
                models = [filepaths[i:i+12] for i in range(0, len(filepaths), 12)]
                # Take deterministic subset using fixed seed
                with torch.random.fork_rng():
                    torch.manual_seed(42)  # Fixed seed for reproducibility
                    selected_models = models[:models_per_class]

                # Flatten the list of selected models
                for model_views in selected_models:
                    new_filepaths.extend(model_views)
        else:
            # For test/comp datasets, use deterministic subset
            model_indices = list(range(total_models))[:subset_size]

            new_filepaths = []
            for idx in model_indices:
                start = idx * 12
                end = (idx + 1) * 12
                new_filepaths.extend(dataset.filepaths[start:end])

        # Update dataset with new filepaths
        dataset.filepaths = new_filepaths

        # Create and cache the dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=train_dataset,  # Only shuffle training data
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between iterations
            prefetch_factor=3       # Prefetch more batches
        )

        self._clear_memory()
        return dataloader

    def validate_model(self, model) -> float:
        # dataset = self.get_dataset(train_dataset=False, test_dataset=True)

        dataset = self.train_loader
        
        if dataset is False:
            raise RuntimeError("Dataset not enough, cannot validate model")

        all_correct = 0
        all_points = 0

        model = model.to(self.device)
        model.eval()
        with torch.inference_mode():
            for data in dataset:
                input = data[1].to(self.device)
                labels = data[0].to(self.device)

                output = model(input)

                aaaaah = torch.max(output, 1)
                # print(len(aaaaah))
                predicted = aaaaah[1]

                results = predicted==labels
                
                all_correct += results.sum().item()
                all_points += len(labels)

        self._clear_memory()

        return float((all_correct / all_points)*100)
    
    def get_comp_time(self, model) -> float:
        dataset = self.get_dataset(train_dataset=False, comp_time_dataset=True)
        
        if dataset is False:
            raise RuntimeError("Dataset not enough, cannot validate model")
        
        model = model.to('cpu')
        # print("Model moved to CPU for inference time calculation")
        model.eval()
        total_time = float(0)
        with torch.inference_mode():
            for data in dataset:
                input = data[1].to('cpu')

                t1 = time.time()
                _ = model(input)
                t2 = time.time()
                total_time += (t2 - t1)

        total_time /= len(dataset)
        # print(f"Average inference time: {total_time} seconds")
        self._clear_memory()
        return total_time


    def train_model(self, model, rank_filter=False, pruner=None) -> torch.nn.Module:
        model = model.train()
        model = model.to(self.device)
        
        if rank_filter:
            print("Getting Ranks")

        while True:
            dataloader = self.get_dataset(train_dataset=True)
            if dataloader is False:
                print("Dataset not enough, trying again")
                continue
            break

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda') if self.device == 'cuda' else None
        running_loss, running_acc, total_steps = 0.0, 0.0, 0.0
        
        for batch_idx, data in enumerate(dataloader):
            with torch.autograd.set_grad_enabled(True):
                inputs = data[1].to(self.device)
                labels = data[0].to(self.device)

                optimizer.zero_grad()
                if rank_filter:
                    pruner.reset()
                    
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = pruner.forward(inputs) if rank_filter else model(inputs)
                        loss = loss_fn(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = pruner.forward(inputs) if rank_filter else model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                
                aaaaah = torch.max(outputs, 1)
                # print(len(aaaaah))
                predicted = aaaaah[1]
                running_acc += (predicted == labels).sum().item()
                total_steps += len(labels)
                if batch_idx % 100 == 0:
                    # print(f"Batch {batch_idx}, Loss: {running_loss / total_steps}, Accuracy: {(running_acc*100) / total_steps}")
                    self._clear_memory()

        # print(f"Training loss: {running_loss / total_steps}, Training accuracy: {(100*running_acc) / total_steps}")
        self._clear_memory()
        return model
    
    def fine_tune(self, model):
        print('*' * 25, "Fine-tuning model", '*' * 25)
        model = model.to(self.device)
        
        # Use a more efficient optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Use learning rate scheduler for faster convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        
        # Early stopping with smarter threshold
        early_stop_count = 0
        early_stop_threshold = 3
        best_accuracy = 0
        
        for epoch in range(10):
            print(f"Fine-tuning epoch {epoch}")
            model = self.train_model(model, optimizer=optimizer)
            
            accuracy = self.validate_model(model)
            scheduler.step(accuracy)
            
            print(f"Validation accuracy for epoch {epoch}: {accuracy}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            # Stop early if no improvement
            if early_stop_count >= early_stop_threshold:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Clear memory each epoch
            self._clear_memory()
        
        return model
    
    def _get_pruning_amounts(self, start, end, step):
        pruning_amounts = np.arange(start, end, step)
        center = np.mean(pruning_amounts)

        sigma = 15  # controls how "tight" around center you prefer; adjust if needed
        probabilities = np.exp(-0.5 * ((pruning_amounts - center) / sigma) ** 2)
        probabilities /= probabilities.sum()  # normalize to sum to 1

        return np.random.choice(
            pruning_amounts,
            size=len(pruning_amounts),
            replace=False,
            p=probabilities,
        )
        
    def write_to_file(self, pruning_amount, pre_acc, post_acc, comp_time, size):
        if 'test_results.txt' not in os.listdir():
            with open('test_results.txt', 'w') as f:
                f.write("Pruning Amount,Pre-Acc,Post-Acc,Comp Time,Model Size\n")
        with open('test_results.txt', 'a') as f:
            f.write(f"{pruning_amount},{pre_acc},{post_acc},{comp_time},{size}\n")
        
    def main(self):        
        pruning_amounts = self._get_pruning_amounts(start=0, end=100, step=4)
        print(pruning_amounts)
        for pruning_amount in pruning_amounts:
            model = self.get_model()
            
            print(f"Pruning amount: {pruning_amount}")
            pruner = PruningFineTuner(model)
            model = pruner.prune(pruning_amount, True)
            
            model_size = self.get_size(model)
            
            print(f"Model size after {pruning_amount}% pruning: {model_size} MB")
            
            pre_acc = self.validate_model(model)
            model = self.fine_tune(model)
            post_acc = self.validate_model(model)
            comp_time = self.get_comp_time(model)
            
            print(f"Pruning Amount {pruning_amount}, Model Size: {model_size}, Pre-Acc: {pre_acc}, Post-Acc: {post_acc}, Comp Time: {comp_time}")
            self.write_to_file(pruning_amount, pre_acc, post_acc, comp_time, model_size)
            self._clear_memory()
            del model
            del pruner

if __name__ == "__main__":
    testing = Testing()
    testing.main()