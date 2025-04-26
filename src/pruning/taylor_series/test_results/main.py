import time
import numpy as np
from MVCNN.models import MVCNN
from MVCNN.tools.ImgDataset import SingleImgDataset
import gc
import torch
from FilterPruner import FilterPruner

class Testing:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_model(self):
        model = MVCNN.SVCNN(
            'svcnn',
        )
        model.load_state_dict(torch.load('model-00030.pth', map_location=self.device))
        model = model.to(self.device)

        return model

    def _clear_memory(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()
        gc.collect()

    def get_dataset(self, train_dataset=True, test_dataset=False, comp_time_dataset=False):
        
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

        subset_size = 0.2 if train_dataset else 0.1 if test_dataset else 0.01 if comp_time_dataset else 0.001
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
                class_name = file_path.split('/')[3]
                if class_name not in classes_present:
                    classes_present.append(class_name)

        if len(classes_present) < 33:
            print(f"Classes not enough: {len(classes_present)}")
            return False
        else:
            print(f"Classes enough: {len(classes_present)}")
            
        self._clear_memory()

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
        )

    def validate_model(self, model):
        dataset = self.get_dataset(train_dataset=False, test_dataset=True)
        all_correct = 0
        all_points = 0
        wrong_class, samples_class = np.zeros(33), np.zeros(33)

        model = model.to(self.device)
        model.eval()

        for _, data in enumerate(dataset):
            input = data[1].to(self.device)
            labels = data[0].to(self.device)

            output = model(input)

            _, predicted = torch.max(output, 1)[1]

            results = predicted==labels

            for i in range(len(results)[0]):
                if not bool(results[i].cpu().numpy()):
                    wrong_class[int(labels[i].cpu().numpy())] += 1
                else:
                    all_correct += 1
                    all_points += 1
                samples_class[int(labels[i].cpu().numpy())] += 1
        all_points += len(results)
        
        self._clear_memory()
        
        return float((all_correct / all_points)*100)
    
    def get_comp_time(self, model):
        dataset = self.get_dataset(train_dataset=False, comp_time_dataset=True)
        model = model.to('cpu')
        print("Model moved to CPU for inference time calculation")
        model.eval()
        total_time = float(0)
        for data in dataset:
            input = data[1].to('cpu')

            t1 = time.time()
            _ = model(input)
            t2 = time.time()
            total_time += (t2 - t1)

        total_time /= len(dataset)
        print(f"Average inference time: {total_time} seconds")
        self._clear_memory()
        return total_time


    def train_model(self, model, rank_filter=False, pruner=None):
        model = model.train()
        model = model.to(self.device)

        dataloader = self.get_dataset(train_dataset=True)
        if dataloader is False:
            print("Dataset not enough, trying again")
            dataloader = self.get_dataset(train_dataset=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        running_loss, running_accc, total_steps = 0.0, 0.0, 0.0
        
        for batch_idx, data in enumerate(dataloader):
            try:
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
                    _, predicted = torch.max(outputs, 1)[1]
                    running_accc += (predicted == labels).sum().item()
                    total_steps += len(labels)
            except Exception as e:
                print(f"Error during training batch {batch_idx}: {e}")
                continue
        print(f"Training loss: {running_loss / total_steps}, Training accuracy: {running_accc / total_steps}")
        self._clear_memory()
        return model
    
    def fine_tune(self, model):
        print("Fine-tuning model...")
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
    
    def get_candidates_to_prune(self, total_filters_to_prune):
        # sourcery skip: class-extract-method
        pruner = FilterPruner(self.get_model())
        pruner.reset()
        self.train_model(pruner.model, rank_filter=True, pruner=pruner)
        pruner.normalize_ranks_per_layer()
        filters_to_prune = pruner.get_pruning_plan(total_filters_to_prune)
        print(f"Filters to prune: {filters_to_prune}")
        self._clear_memory()
        return filters_to_prune
    
    def get_total_filters(self):
        return sum(
            layer.out_channels
            for layer in self.get_model().net_1
            if isinstance(layer, torch.nn.Conv2d)
        )
    
    def get_model_size(self, model):
        total_size = sum(
            param.nelement() * param.element_size()
            for param in model.parameters()
        )
        return total_size / (1024 ** 2)
    
    def prune(self, pruning_amount, ranks):
        total_filters = self.get_total_filters()
        filters_to_prune = int(total_filters * pruning_amount)
        
        print(f"Total filters: {total_filters}, Filters to prune: {filters_to_prune}")
        print(f"Pruning {pruning_amount*100}% of filters")
        
        if len(ranks) > 0:
            prune_targets = ranks
            # take only x amount of filters to prune
            prune_targets = prune_targets[:filters_to_prune]
            prune_targets = [(layer_index, filter_index) for layer_index, filter_index, _ in prune_targets]
        else:
            prune_targets = self.get_candidates_to_prune(filters_to_prune)
        layers_pruned = {}
        for layer_index, filter_index in prune_targets:
            layers_pruned[layer_index] = layers_pruned.get(layer_index, 0) + 1
        print(f"Layers pruned: {layers_pruned}")
        
        print("Pruning Filters")
        
        model = self.get_model()
        pruner = FilterPruner(model)
        
        for idx, (layer_index, filter_index) in enumerate(prune_targets):
            model = pruner.prune_vgg_conv_layer(model, layer_index, filter_index)
            
            if idx % 10 == 0:
                print(f"Pruned {idx} filters")
                self._clear_memory()
        
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float().to(self.device)
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float().to(self.device)
        print("Model pruned")
        return model
    
    def get_ranks(self, model):
        pruner = FilterPruner(model)
        pruner.reset()
        self.train_model(pruner.model, rank_filter=True, pruner=pruner)
        pruner.normalize_ranks_per_layer()
        ranks = pruner.get_pruning_plan(0)
        print(f"Length of Ranks: {len(ranks)}")
        del pruner
        self._clear_memory()
        
        if len(ranks) > 0:
            raise ValueError("Ranks are empty, something went wrong")
        else:
            return ranks
    
    def _init_csv(self, filename=str(time.time())):
        with open(filename, 'w') as f:
            f.write("pruning_amount,pre_accuracy, post_accuracy, comp_time, model_size\n")
        print(f"CSV file {filename} initialized")
    
    def _write_csv(self, filename, pruning_amount, pre_accuracy, post_accuracy, comp_time, model_size):
        with open(filename, 'a') as f:
            f.write(f"{pruning_amount},{pre_accuracy},{post_accuracy},{comp_time},{model_size}\n")
        print(f"Data written to {filename}")
        
    def run(self):
        print("Starting testing...")
        pruning_amounts =  np.arange(0.0, 1.0, 0.005)
        # 0% to 100%, 1% increments
        print(f"Pruning amounts: {pruning_amounts}")
        
        ranks = self.get_ranks(self.get_model())
        
        filename = f"results_{int(time.time())}.csv"
        self._init_csv(filename)
        
        for pruning_amount in pruning_amounts:
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