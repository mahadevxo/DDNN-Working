import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm

class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12, device='cuda'):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)


    def train(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            
            # Initialize tqdm progress bar
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            running_loss = 0.0
            running_acc = 0.0
            total_steps = 0
            
            for i, data in enumerate(pbar):

                if self.model_name == 'mvcnn':
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).to(self.device)  # Reshape to (batch_size * num_views, C, H, W)
                    target = Variable(data[0]).to(self.device).repeat_interleave(V)  # Repeat target for each view
                else:
                    in_data = Variable(data[1]).to(self.device)
                    target = Variable(data[0]).to(self.device)

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)
                running_loss += loss.item()

                self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]
                running_acc += acc.item()
                total_steps += 1
                
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc+i+1)

                loss.backward()
                self.optimizer.step()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{running_loss/total_steps:.3f}", 
                    'acc': f"{running_acc/total_steps:.3f}"
                })
            
            # Print final statistics for the epoch
            print(f"Epoch {epoch+1}/{n_epochs} - Avg Loss: {running_loss/total_steps:.3f}, Avg Accuracy: {running_acc/total_steps:.3f}")
            
            i_acc += i

            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
                self.writer.add_scalar('val/val_loss', loss, epoch+1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)

            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

            # Free up unused GPU memory at the end of each epoch
            torch.cuda.empty_cache()

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(f"{self.log_dir}/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0
        all_loss = 0
        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)

        self.model.eval()

        # Add progress bar for validation
        val_pbar = tqdm(self.val_loader, desc=f"Validation - Epoch {epoch+1}")
        
        for _, data in enumerate(val_pbar, 0):
            if self.model_name == 'mvcnn':
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(-1, C, H, W).to(self.device)
                # Fix: Repeat target for each view to match the batch size of in_data
                target = Variable(data[0]).to(self.device).repeat_interleave(V)
            else: # 'svcnn'
                in_data = Variable(data[1]).to(self.device)
                target = Variable(data[0]).to(self.device)
                
            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target
            
            # For class accuracy calculation, we need to handle repeated targets for MVCNN
            if self.model_name == 'mvcnn':
                # Only count each object once by taking every V-th prediction
                for i in range(N):
                    obj_preds = pred[i*V:(i+1)*V]
                    # Use majority voting across views
                    obj_pred = torch.mode(obj_preds)[0]
                    obj_target = target[i*V]  # All targets for the same object are identical
                    
                    if obj_pred != obj_target:
                        wrong_class[obj_target.cpu().data.numpy().astype('int')] += 1
                    samples_class[obj_target.cpu().data.numpy().astype('int')] += 1
            else:
                for i in range(results.size()[0]):
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                    samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            
            correct_points = torch.sum(results.long())
            all_correct_points += correct_points
            all_points += results.size()[0]
            
            # Update progress bar with running validation accuracy
            current_acc = correct_points.float() / results.size()[0]
            val_pbar.set_postfix({'acc': f"{current_acc.item():.3f}"})
            
        # Calculate final metrics
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        val_overall_acc = all_correct_points.float() / all_points
        val_overall_acc = val_overall_acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)
        
        # Print final validation results in a clean format
        print(f"Validation Results - Epoch {epoch+1}:")
        print(f"  Total samples: {all_points}")
        print(f"  Mean Class Accuracy: {val_mean_class_acc:.4f}")
        print(f"  Overall Accuracy: {val_overall_acc:.4f}")
        print(f"  Loss: {loss:.4f}")
        print("-" * 50)
        
        self.model.train()
        return loss, val_overall_acc, val_mean_class_acc

