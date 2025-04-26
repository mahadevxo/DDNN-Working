import time
import numpy as np
from MVCNN.models import MVCNN
from MVCNN.tools.ImgDataset import SingleImgDataset
import gc
import torch
from FilterPruner import FilterPruner
from Pruning import Pruning

#TODO find why VGG11 and SVCNN are acting differently
#TODO add a way to get the model size
#TODO fix the pruning amount accuracy etc code
class Testing:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_model(self) -> torch.nn.Module:
        """
        Loads a pre-trained SVCNN model from disk.

        The model is loaded into the device specified in the constructor.

        Returns:
            A pre-trained SVCNN model.
        """
        model = MVCNN.SVCNN(
            'svcnn',
        )
        model.load_state_dict(torch.load('model-00030.pth', map_location=self.device))
        model = model.to(self.device)

        return model

    def _clear_memory(self) -> None:
        """
        Clears the memory cache on the current device and collects garbage.

        This method empties the cache for the CUDA or MPS device if applicable,
        and performs garbage collection to free up memory resources.
        """

        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()
        gc.collect()

    def get_dataset(self, train_dataset=True, test_dataset=False, comp_time_dataset=False) -> torch.utils.data.DataLoader:
        
        """
        Loads and returns a DataLoader containing a subset of the ModelNet40 dataset.

        The subset includes 20% of the models in the train dataset, 100% of the models
        in the test dataset, and 1% of the models in the comp time dataset.

        The dataset is loaded from a directory tree with the following structure:
        root_dir/*
            train/*
                class1/*
                    model1_view1.png
                    model1_view2.png
                    ...
                class2/*
                    model2_view1.png
                    model2_view2.png
                    ...
            test/*
                class1/*
                    model1_view1.png
                    model1_view2.png
                    ...
                class2/*
                    model2_view1.png
                    model2_view2.png
                    ...

        The root directory can be specified using the root_dir argument.

        Args:
            train_dataset (bool): Whether to load the train dataset.
            test_dataset (bool): Whether to load the test dataset.
            comp_time_dataset (bool): Whether to load a subset of the test dataset
                for comparison of inference times.

        Returns:
            A DataLoader object containing the subset of the ModelNet40 dataset.
        """
        if train_dataset:
            dataset = SingleImgDataset(
                root_dir='ModelNet40-12View/*/train',
                scale_aug=False,
                rot_aug=False,
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
        
        # if test_dataset:
        #     print(f"Total models in test dataset: {total_models}")
        # if comp_time_dataset:
        #     print(f"Total models in comp time dataset: {total_models}")
        # if train_dataset:
        #     print(f"Total models in train dataset: {total_models}")

        subset_size = 0.2 if train_dataset else 1.0 if test_dataset else 0.01 if comp_time_dataset else None
        
        if subset_size is None:
            raise ValueError("Invalid subset size")
        subset_size = int(total_models * subset_size)

        rand_model_indices = torch.randperm(total_models)[:subset_size]
        rand_model_indices = rand_model_indices.tolist()

        new_filepaths = []
        for idx in rand_model_indices:
            start = idx * 12
            end = (idx+1) * 12

            new_filepaths.extend(dataset.filepaths[start:end])

        dataset.filepaths = new_filepaths
        classes_present = []
        if train_dataset:
            for file_path in dataset.filepaths:
                class_name = file_path.split('/')[1]
                if class_name not in classes_present:
                    classes_present.append(class_name)
        
        classes_present = set(classes_present)
        
        if train_dataset and len(classes_present) < 33:
            print(f"Classes not enough: {len(classes_present)}")
            return False

        self._clear_memory()

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
        )

    def validate_model(self, model) -> float:
        """
        Validates the given model using the test dataset and returns the accuracy as a percentage.

        This method retrieves the test dataset and runs the model in evaluation mode on it.
        The model's predictions are compared to the true labels to calculate the number of correct 
        predictions. The accuracy is computed as the percentage of correct predictions out of the 
        total number of samples in the dataset.

        Args:
            model (torch.nn.Module): The model to be validated.

        Returns:
            float: The accuracy of the model on the test dataset, expressed as a percentage.

        Raises:
            RuntimeError: If the test dataset is not sufficient for validation.
        """

        dataset = self.get_dataset(train_dataset=False, test_dataset=True)

        if dataset is False:
            raise RuntimeError("Dataset not enough, cannot validate model")

        all_correct = 0
        all_points = 0

        model = model.to(self.device)
        model.eval()

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
        """
        Calculates the average inference time of the given model on a subset of the test dataset.

        This method retrieves the subset of the test dataset and runs the model in evaluation mode on it.
        The time taken to run the model on each sample is recorded and an average is calculated.

        Args:
            model (torch.nn.Module): The model to calculate the average inference time for.

        Returns:
            float: The average inference time of the model on the subset of the test dataset, in seconds.

        Raises:
            RuntimeError: If the subset of the test dataset is not sufficient for calculation of inference time.
        """
        dataset = self.get_dataset(train_dataset=False, comp_time_dataset=True)
        
        if dataset is False:
            raise RuntimeError("Dataset not enough, cannot validate model")
        
        model = model.to('cpu')
        # print("Model moved to CPU for inference time calculation")
        model.eval()
        total_time = float(0)
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
        """
        Train a model on the dataset.

        Args:
            model (torch.nn.Module): The model to train.
            rank_filter (bool, optional): Whether to use rank filtering. Defaults to False.
            pruner (FilterPruner, optional): The filter pruner to use. Defaults to None.

        Returns:
            torch.nn.Module: The trained model.
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
        
        running_loss, running_acc, total_steps = 0.0, 0.0, 0.0
        
        for batch_idx, data in enumerate(dataloader):
            with torch.autograd.set_grad_enabled(True):
                inputs = data[1].to(self.device)
                labels = data[0].to(self.device)

                optimizer.zero_grad()
                if rank_filter:
                    pruner.reset()
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
    
    def fine_tune(self, model) -> torch.nn.Module:
        """
        Fine-tunes a given model on the dataset.

        This function performs a few epochs of training on a given model using SGD optimizer and
        cross-entropy loss to adapt the model to the changes after pruning.

        Args:
            model (torch.nn.Module): The model to fine-tune.

        Returns:
            The fine-tuned model.
        """
        print('*' * 25, "Fine-tuning model", '*' * 25)
        model = model.to(self.device)
        
        epoch = 0
        prev_accs = []
        best_accuracy = 0
        
        while True:
            print(f"Fine-tuning epoch {epoch}")
            model = self.train_model(model)
            
            accuracy = self.validate_model(model)
            print(f"Validation accuracy for epoch {epoch}: {accuracy}")
            prev_accs.append(accuracy)
            if len(prev_accs) > 5:
                prev_accs.pop(0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            if epoch > 5 and best_accuracy - 2 <= np.mean(prev_accs).item() <= best_accuracy + 2:
                print("No Improvement in accuracy; best accuracy:", best_accuracy, "Mean accuracy:", np.mean(prev_accs).item())
                break
            
            if epoch > 7:
                print("Stopping fine-tuning...")
                print(f"Best accuracy: {best_accuracy}, Mean accuracy: {np.mean(prev_accs).item()}")
                break
            
            epoch += 1
        
        self._clear_memory()
        
        return model
    
    def get_candidates_to_prune(self, total_filters_to_prune) -> list:
        # sourcery skip: class-extract-method
        pruner = FilterPruner(self.get_model())
        pruner.reset()
        self.train_model(pruner.model, rank_filter=True, pruner=pruner)
        pruner.normalize_ranks_per_layer()
        filters_to_prune = pruner.get_pruning_plan(total_filters_to_prune)
        print(f"Filters to prune: {filters_to_prune}")
        self._clear_memory()
        return filters_to_prune
    
    def get_total_filters(self) -> int:
        return sum(
            layer.out_channels
            for layer in self.get_model().net_1
            if isinstance(layer, torch.nn.Conv2d)
        )
    
    def get_model_size(self, model) -> float:
        total_size = sum(
            param.nelement() * param.element_size()
            for param in model.parameters()
        )
        return total_size / (1024 ** 2)
    
    def prune(self, pruning_amount, ranks) -> torch.nn.Module:
        # sourcery skip: low-code-quality
        total_filters = self.get_total_filters()
        filters_to_prune = int(total_filters * pruning_amount)
        model_size_before = self.get_model_size(self.get_model())

        print(f"Total filters: {total_filters}, Filters to prune: {filters_to_prune} ({pruning_amount*100:.1f}%)")

        if len(ranks) > 0:
            prune_targets = ranks[:filters_to_prune]
        else:
            prune_targets = self.get_candidates_to_prune(filters_to_prune)

        print('*' * 10, "Pruning Filters", '*' * 10)
        model = self.get_model()

        # Create mapping of conv layers to their indices in module list
        conv_layer_mapping = {}
        for i, layer in enumerate(model.net_1):
            if isinstance(layer, torch.nn.Conv2d):
                # Use a different index sequence for the pruning target lookup
                # Filter pruner and conv layer indices might be different
                idx = len(conv_layer_mapping)
                conv_layer_mapping[idx] = i

        print(f"Conv layer mapping: {conv_layer_mapping}")

        # Track channel counts before pruning
        channels_per_layer = {}
        for i in conv_layer_mapping.values():
            layer = model.net_1[i]
            if isinstance(layer, torch.nn.Conv2d):
                channels_per_layer[i] = layer.out_channels

        # Filter pruning targets to avoid pruning layers with only 1 channel
        filtered_prune_targets = []
        for filter_index_from_rank, filter_num in prune_targets:
            # Convert the filter pruner index to the actual layer index in model.net_1
            if filter_index_from_rank not in conv_layer_mapping:
                print(f"Filter index {filter_index_from_rank} not found in mapping, skipping")
                continue

            actual_layer_index = conv_layer_mapping[filter_index_from_rank]

            if actual_layer_index not in channels_per_layer:
                print(f"Layer {actual_layer_index} not found in channels count")
                continue

            if channels_per_layer[actual_layer_index] <= 1:
                print(f"Layer {actual_layer_index} has only 1 channel, skipping pruning")
                continue

            channels_per_layer[actual_layer_index] -= 1
            filtered_prune_targets.append((actual_layer_index, filter_num))

        print(f"Filters to prune after filtering: {len(filtered_prune_targets)}")

        # Add debug to examine the actual structure
        if not filtered_prune_targets:
            print("No filters to prune! Examining model structure...")
            for i, layer in enumerate(model.net_1):
                if isinstance(layer, torch.nn.Conv2d):
                    print(f"Layer {i}: Conv2d with {layer.out_channels} output channels")

        pruner = Pruning(model)

        # Track pruning progress
        pruned_filters = 0
        for idx, (layer_index, filter_index) in enumerate(filtered_prune_targets):
            try:
                model = pruner.prune_vgg_conv_layer(model, layer_index, filter_index)
                pruned_filters += 1

                if idx % 50 == 0 and idx > 0:
                    print(f"Pruned {idx}/{len(filtered_prune_targets)} filters")
                    self._clear_memory()

            except Exception as e:
                print(f"Error pruning layer {layer_index}, filter {filter_index}: {e}")
                import traceback
                traceback.print_exc()

        # Verify model changes
        model_size_after = self.get_model_size(model)
        size_reduction = (model_size_before - model_size_after) / model_size_before * 100

        print("Pruning summary:")
        print(f"- Filters pruned: {pruned_filters}/{len(filtered_prune_targets)} attempted")
        print(f"- Model size: {model_size_before:.2f} MB → {model_size_after:.2f} MB ({size_reduction:.1f}% reduction)")

        return model
    
    def get_ranks(self, model) -> list:
        pruner = FilterPruner(model)
        pruner.reset()
        self.train_model(pruner.model, rank_filter=True, pruner=pruner)
        total_filters = self.get_total_filters()
        pruner.normalize_ranks_per_layer()
        ranks = pruner.get_pruning_plan(total_filters)
        print(f"Length of Ranks: {len(ranks)}")
        del pruner
        self._clear_memory()
        
        if len(ranks) == 0:
            raise ValueError("Ranks are empty, something went wrong")
        else:
            return ranks
    
    def _init_csv(self, filename=str(time.time())) -> None:
        with open(filename, 'w') as f:
            f.write("pruning_amount,pre_accuracy, post_accuracy, comp_time, model_size\n")
        print(f"CSV file {filename} initialized")
    
    def _write_csv(self, filename, pruning_amount, pre_accuracy, post_accuracy, comp_time, model_size) -> None:
        with open(filename, 'a') as f:
            f.write(f"{pruning_amount},{pre_accuracy},{post_accuracy},{comp_time},{model_size}\n")
        print(f"Data written to {filename}")
        
    def run(self) -> None:
        print("Starting testing...")
        pruning_amounts =  np.arange(0.0, 1.0, 0.005)
        #randomize it
        np.random.shuffle(pruning_amounts)
        # 0% to 100%, 1% increments
        print(f"Total Pruning amounts: {len(pruning_amounts)}")
        
        ranks = self.get_ranks(self.get_model())
        
        filename = f"results_{int(time.time())}.csv"
        self._init_csv(filename)
        
        for pruning_amount in pruning_amounts:
            print('*' * 25, f"Pruning Amount: {pruning_amount}", '*' * 25)
            
            model = self.prune(pruning_amount, ranks)
            
            model_size = self.get_model_size(model)
            print(f"Model size after pruning: {model_size} MB")
            
            pre_accuracy = self.validate_model(model)
            print(f"Validation accuracy after pruning: {pre_accuracy}")
            
            comp_time = self.get_comp_time(model)
            print(f"Computation time after pruning: {comp_time} seconds")
            
            model = self.fine_tune(model)
            
            post_accuracy = self.validate_model(model)
            print(f"Validation accuracy after fine-tuning: {post_accuracy}")
            
            self._write_csv(
                filename,
                pruning_amount,
                pre_accuracy,
                post_accuracy,
                comp_time,
                model_size
            )
            self._clear_memory()
        print("Testing completed")
        print(f"Results saved to {filename}")
        
if __name__ == "__main__":
    tester = Testing()
    tester.run()