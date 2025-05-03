from copy import deepcopy
import gc
import time
import torch
from torch.utils.data import DataLoader
import resource
from MVCNN.tools.ImgDataset import SingleImgDataset
from MVCNN.models import MVCNN
from PFT import PruningFineTuner

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
print(f"File descriptor limit increased from {soft} to {hard}")

class InfoGetter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.train_path = 'ModelNet40-12View/*/train'
        self.test_path = 'ModelNet40-12View/*/test'
        self.criterion = torch.nn.CrossEntropyLoss()
        self._clear_memory()
        
        test_dataset = SingleImgDataset(
            root_dir=self.test_path,
            scale_aug=False,
            rot_aug=False,
            num_models=1000,
            num_views=12,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
    def _clear_memory(self):
        """
        Clears GPU/MPS memory by forcing garbage collection and emptying caches.
        
        Helps prevent memory leaks by explicitly freeing unused memory
        after operations that might create large temporary tensors.
        
        Args:
            None
            
        Returns:
            None
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def get_model(self):
        """
        Loads and returns a pre-trained model.
        
        Creates an SVCNN model instance and loads pre-trained weights from a file,
        then returns a deep copy of the model.
        
        Args:
            None
            
        Returns:
            Deep copy of the loaded model with pre-trained weights
        """
        model = MVCNN.SVCNN(
            'svcnn'
        )
        model.load_state_dict(torch.load('model-00030.pth', map_location=self.device))
        model = model.to(self.device)
        
        return deepcopy(model)
    
    def get_dataset(self, train_dataset=False, test_dataset=False, comp_time_dataset=False):
        """
        Creates appropriate datasets for different evaluation needs.
        
        Creates and returns DataLoaders with different configurations based on
        whether they're needed for training, testing, or computation time evaluation.
        
        Args:
            train_dataset: If True, returns a dataset for training
            test_dataset: If True, returns a dataset for testing
            comp_time_dataset: If True, returns a dataset for computation time evaluation
            
        Returns:
            DataLoader configured for the requested purpose
        """
        # Load dataset
        if train_dataset:
            dataset = SingleImgDataset(
                root_dir=self.train_path,
                scale_aug=False,
                rot_aug=False,
                num_models=1000,
                num_views=12,
            )
        elif comp_time_dataset:
            dataset = SingleImgDataset(
                root_dir=self.test_path,
                scale_aug=False,
                rot_aug=False,
                num_models=1000,
                num_views=12,
            )
        elif test_dataset:
            return self.test_loader
        else:
            raise ValueError("Invalid dataset type specified.")
        
        total_models = len(dataset) // 12
        
        subset_size = 0.1 if train_dataset else 0.03 if comp_time_dataset else None
        
        if subset_size is None:
            raise ValueError("Invalid subset size specified.")
        
        subset_size = int(total_models * subset_size)
        
        if train_dataset:
            class_filepaths = {}
            for filepath in dataset.filepaths:
                class_name = filepath.split('/')[1]
                if class_name not in class_filepaths:
                    class_filepaths[class_name] = []
                class_filepaths[class_name].append(filepath)
            
            if len(class_filepaths) < 33:
                print(f"Not enough classes in the dataset, got {len(class_filepaths)} classes")
                return False

            models_per_class = max(1, subset_size // len(class_filepaths))
            new_filepaths = []
            
            for filepaths in class_filepaths.values():
                models = [filepaths[i:i+12] for i in range(0, len(filepaths), 12)]
                
                with torch.random.fork_rng():
                    selected_models = models[:models_per_class]

                for model_views in selected_models:
                    new_filepaths.extend(model_views)
        
        elif comp_time_dataset:
            model_indices = list(range(total_models))[:subset_size]
            
            new_filepaths = []
            for idx in model_indices:
                start = idx * 12
                end = (idx+1) * 12
                new_filepaths.extend(dataset.filepaths[start:end])
        
        dataset.filepaths = new_filepaths
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=8,
            shuffle=train_dataset,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2
        )
        
        self._clear_memory()
        return dataloader
    
    def get_accuracy(self, model):
        """
        Evaluates the model's accuracy on the test dataset.
        
        Runs the model on the test dataset and calculates the percentage
        of correct predictions.
        
        Args:
            model: The model to evaluate
            
        Returns:
            Float representing the accuracy percentage
        """
        model.eval()
        
        test_loader = self.get_dataset(test_dataset=True)
        
        if test_loader is False:
            raise RuntimeError("Not enough classes in the dataset")
        
        correct = 0
        total = 0
        
        model = model.to(self.device)
        model.eval()
        with torch.inference_mode():
            for data in test_loader:
                input = data[1].to(self.device)
                labels = data[0].to(self.device)
                
                outputs = model(input)
                
                predicted = torch.max(outputs, 1)[1]
                
                results = (predicted == labels)
                
                correct += results.sum().item()
                
                total += len(labels)
        
        accuracy = float((correct/total)*100) # accuracy in percentage
        self._clear_memory()
        return accuracy

    def get_size(self, model):
        """
        Calculates the size of the model in megabytes.
        
        Sums the memory usage of all parameters and buffers in the model
        to determine the total model size in MB.
        
        Args:
            model: The model whose size to calculate
            
        Returns:
            Float representing the model size in MB
        """
        param_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        buffer_size = sum(
            buffer.nelement() * buffer.element_size() for buffer in model.buffers()
        )
        return (param_size + buffer_size) / (1024 ** 2)
    
    def get_comp_time(self, model):
        """
        Measures the average inference time of the model.
        
        Runs the model on a subset of the dataset multiple times and calculates
        the average computation time per batch.
        
        Args:
            model: The model whose computation time to measure
            
        Returns:
            Float representing average inference time per batch
        """
        dataset = self.get_dataset(comp_time_dataset=True)
        
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
    
    def train_model(self, model, rank_filter=False, pruner=None):
        """
        Trains a model on a subset of the training data.
        
        If rank_filter is True, uses the pruner to compute filter importance scores
        during training. Otherwise, performs regular training to update model weights.
        
        Args:
            model: The model to train
            rank_filter: If True, computes filter rankings instead of regular training
            pruner: FilterPruner instance for computing rankings (required if rank_filter=True)
            
        Returns:
            The trained model
        """
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
                
                predicted = torch.max(outputs, 1)[1]
                
                running_acc += (predicted == labels).sum().item()
                total_steps += len(labels)
                if batch_idx % 100 == 0:
                    # print(f"Batch {batch_idx}, Loss: {running_loss / total_steps}, Accuracy: {(running_acc*100) / total_steps}")
                    self._clear_memory()

        # print(f"Training loss: {running_loss / total_steps}, Training accuracy: {(100*running_acc) / total_steps}")
        self._clear_memory()
        return model
    
    def fine_tune(self, model):
        """
        Fine-tunes a pruned model to recover accuracy.
        
        Performs several epochs of training with adaptive learning rate and
        early stopping to efficiently fine-tune a pruned model.
        
        Args:
            model: The pruned model to fine-tune
            
        Returns:
            The fine-tuned model
        """
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
        early_stop_threshold = 5
        best_accuracy = 0
        
        for epoch in range(10):
            print(f"Fine-tuning epoch {epoch}")
            model = self.train_model(model)
            
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
    
    def getInfo(self, pruning_amount):
        """
        Gets comprehensive information about a model pruned by the specified amount.
        
        Prunes the model, fine-tunes it, and evaluates its accuracy, computation time,
        and size to provide a complete picture of model performance at the given
        pruning level.
        
        Args:
            pruning_amount: Percentage of filters to prune (0-1)
            
        Returns:
            Tuple containing (model, accuracy, computation_time, size)
        """
        model = self.get_model()
        model = model.to(self.device)
        
        pruner = PruningFineTuner(model)
        model = pruner.prune(pruning_amount)
        
        pre_acc = self.get_accuracy(model)
        model = self.fine_tune(model)
        post_acc = self.get_accuracy(model)
        comp_time = self.get_comp_time(model)
        size = self.get_size(model)
        
        print(f"Pruning Amount {pruning_amount}, Model Size: {size}, Pre-Acc: {pre_acc}, Post-Acc: {post_acc}, Comp Time: {comp_time}")
        
        return model, post_acc, comp_time, size