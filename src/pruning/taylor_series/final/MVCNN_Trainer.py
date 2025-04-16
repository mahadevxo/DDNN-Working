import time
import torch
import numpy as np
import gc
from FilterPruner import FilterPruner
from tools.ImgDataset import SingleImgDataset


class MVCNN_Trainer():
    def __init__(self, optimizer = None, num_views=12, train_amt=0.1, test_amt=0.1):
        self.optimizer = optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_views = num_views
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_path = 'ModelNet40-12View/*/train'
        self.test_path = 'ModelNet40-12View/*/test'   
        self.num_models=1000*12
        self.num_views=12
        self.train_amt = train_amt
        self.test_amt = test_amt
    
    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
    def get_train_data(self):
        train_dataset = SingleImgDataset(
            self.train_path, scale_aug=False, rot_aug=False,
            num_models=self.num_models, num_views=self.num_views,
        )
        total_models = int(len(train_dataset.filepaths) / self.num_views)

        # Determine how many models to sample
        subset_size = int(self.train_amt * total_models)

        # Randomly sample model indices
        rand_model_indices = np.random.permutation(total_models)[:subset_size]

        # Gather subset of filepaths (keep model groupings)
        new_filepaths = []
        for idx in rand_model_indices:
            start = idx * self.num_views
            end = (idx + 1) * self.num_views
            new_filepaths.extend(train_dataset.filepaths[start:end])

        # Assign the new filepaths to the dataset
        train_dataset.filepaths = new_filepaths
        print(f'Train Data: {len(new_filepaths)}')

        return torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=4
        )
    
    def get_test_data(self):
        test_dataset = SingleImgDataset(
            self.test_path, scale_aug=False, rot_aug=False,
            num_models=self.num_models, num_views=self.num_views,
        )
        total_models = int(len(test_dataset.filepaths) / self.num_views)

        # Determine how many models to sample
        subset_size = int(self.test_amt * total_models)

        # Randomly sample model indices
        rand_model_indices = np.random.permutation(total_models)[:subset_size]

        # Gather subset of filepaths (keep model groupings)
        new_filepaths = []
        for idx in rand_model_indices:
            start = idx * self.num_views
            end = (idx + 1) * self.num_views
            new_filepaths.extend(test_dataset.filepaths[start:end])

        # Assign the new filepaths to the dataset
        test_dataset.filepaths = new_filepaths
        print(f'Test Data: {len(new_filepaths)}')

        return torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=True, num_workers=4
        )
    
    
    def get_val_accuracy(self, model, test_loader=None):
        all_correct_points = 0
        all_points = 0
        all_loss = 0
        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        
        if test_loader is None:
            test_loader = self.get_test_data()
        model = model.to(self.device)
        model.eval()
        
        for _, data in enumerate(test_loader, 0):
            
            # FOR SVCNN
            in_data = data[1].to(self.device)
            labels = data[0].to(self.device)
            
            t1 = time.time()
            output = model(in_data)
            t2 = time.time()
            time_taken = t2 - t1
            
            pred = torch.max(output, 1)[1]
            all_loss += self.loss_fn(output, labels).cpu().detach().numpy()
            results = pred==labels
            
            for i in range(results.size()[0]):
                if not bool(results[i].cpu().numpy()):
                    wrong_class[labels[i].cpu().numpy()] += 1
                else:
                    all_correct_points += 1
                all_points += 1
                samples_class[labels[i].cpu().numpy()] += 1
        all_loss /= len(test_loader)
        
        val_accuracy = float((all_correct_points / all_points)*100)
        val_class_acc = 1 - (np.nan_to_num(wrong_class) / np.nan_to_num(samples_class))
        val_class_acc = np.mean(val_class_acc)
        val_class_acc = np.nan_to_num(val_class_acc)
        
        model.train()
        return all_loss, val_accuracy, val_class_acc, time_taken
    
    def train_model(self, model, train_loader=None, rank_filter=False, pruner_instance=None):
        model = model.train()
        model = model.to(self.device)

        if train_loader is None:
            train_loader = self.get_train_data()

        running_loss = 0.0
        running_acc = 0.0
        total_steps = 0.0

        # Use provided pruner_instance if ranking filters
        if rank_filter:
            pruner = pruner_instance if pruner_instance is not None else FilterPruner(model)

        for batch_idx, data in enumerate(train_loader):
            try:
                with torch.autograd.set_grad_enabled(True):
                    if self.optimizer is not None:
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        model.zero_grad(set_to_none=True)

                    in_data = data[1].to(self.device)
                    labels = data[0].to(self.device)

                    output = pruner.forward(in_data) if rank_filter else model(in_data)
                    loss = self.loss_fn(output, labels)
                    loss.backward()

                    running_loss += loss.item()

                    if self.optimizer is not None:
                        self.optimizer.step()

                    pred = torch.max(output, 1)[1]
                    results = pred==labels
                    correct_points = torch.sum(results.long())
                    running_acc += correct_points.cpu().numpy()
                    total_steps += 1
            except Exception as exp:
                print(f"Error during training batch {batch_idx}: {exp}")
                continue

        vals = self.get_val_accuracy(model, self.get_test_data())
        # print(f'Train Loss: {running_loss/total_steps}, Train Accuracy: {running_acc/total_steps}, Val Loss: {vals[0]}, Val Accuracy: {vals[1]}')
        self._clear_memory()
        return model
            
    def fine_tune(self, model, rank_filer=False):
        print("Fine Tuning Model")        
        model = model.to(self.device)
        
        _, val_accuracy, _ = self.get_val_accuracy(model, self.get_test_data())
        print(f"Initial Validation Accuracy: {val_accuracy}")
        model = model.train()
        epoch = 0
        prev_accs = []
        best_accuracy = 0
        while True:
            print(f'Fine Tuning Epoch {epoch+1}')
            model = self.train_model(
                model, 
                self.get_train_data,
                rank_filter=rank_filer,
            )
            
            accuracy = self.get_val_accuracy(model, self.get_test_data())[1]
            print(f"Validation Accuracy: {accuracy}")
            prev_accs.append(accuracy)
            
            if len(prev_accs) > 5:
                prev_accs.pop(0)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            if epoch > 3 and best_accuracy -2 <= np.mean(prev_accs).item() <= best_accuracy + 2:
                print(f'No improvement in accuracy; best accuracy {best_accuracy}')
                print(f'Mean accuracy over the last 5 epochs: {np.mean(prev_accs).item()}')
                break
            if epoch >=20:
                print('Max epochs reached')
                break
        
        print("---"*50, "Final Results", "---"*50)
        print(f"Final Validation Accuracy: {accuracy}")
        print(f"Best Validation Accuracy: {best_accuracy}")
        return model, accuracy
    
    def get_size(self, model):
        total_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        return total_size / (1024 ** 2)