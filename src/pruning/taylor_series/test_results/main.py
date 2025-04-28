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
            
            if epoch > 10:
                print("Stopping fine-tuning after 10 epochs...")
                print(f"Best accuracy: {best_accuracy}, Mean accuracy: {np.mean(prev_accs).item()}")
                break
            
            epoch += 1
        
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