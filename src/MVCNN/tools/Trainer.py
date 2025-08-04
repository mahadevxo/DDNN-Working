import torch
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
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

        self.model.to(self.device)
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)
        # Use AMP only if device is cuda
        self.use_amp = (self.device == 'cuda')
        self.scaler = torch.GradScaler() if self.use_amp else None

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

                in_data = data[1].to(self.device)
                target = data[0].to(self.device)

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        out_data = self.model(in_data)
                        loss = self.loss_fn(out_data, target)
                else:
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

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
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
        wrong_class = np.zeros(33)  # Fix: Use 33 for ModelNet33
        samples_class = np.zeros(33)

        self.model.eval()

        # Add progress bar for validation
        val_pbar = tqdm(self.val_loader, desc=f"Validation - Epoch {epoch+1}")

        for _, data in enumerate(val_pbar, 0):
            if self.model_name == 'mvcnn':
                N, V, C, H, W = data[1].size()
                in_data = data[1].view(-1, C, H, W).to(self.device)
                target = data[0].to(self.device).repeat_interleave(V)
            else: # 'svcnn'
                in_data = data[1].to(self.device)
                target = data[0].to(self.device)

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            # Fix: Handle MVCNN and SVCNN differently
            if self.model_name == 'mvcnn':
                # Count object-level accuracy using majority voting
                obj_correct = 0
                for i in range(N):
                    obj_preds = pred[i*V:(i+1)*V]  # All predictions for object i
                    obj_target = target[i*V]  # Target for object i

                    # Majority voting across views
                    obj_pred = torch.mode(obj_preds)[0]

                    # Update class-wise statistics
                    if obj_pred != obj_target:
                        wrong_class[obj_target.cpu().data.numpy().astype('int')] += 1
                    else:
                        obj_correct += 1
                    samples_class[obj_target.cpu().data.numpy().astype('int')] += 1

                # For MVCNN: count objects, not views
                all_correct_points += obj_correct
                all_points += N  # Count objects, not views

                # Update progress bar with object-level accuracy
                current_acc = obj_correct / N

            else: # SVCNN
                results = pred == target

                for i in range(results.size()[0]):
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                    samples_class[target.cpu().data.numpy().astype('int')[i]] += 1

                correct_points = torch.sum(results.long())
                all_correct_points += correct_points
                all_points += results.size()[0]

                # Update progress bar with view-level accuracy
                current_acc = correct_points.float() / results.size()[0]
                current_acc = current_acc.item()

            val_pbar.set_postfix({'acc': f"{current_acc:.3f}"})

        # Calculate final metrics
        print(samples_class)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        val_overall_acc = all_correct_points / all_points

        # Convert to numpy if it's a tensor
        if isinstance(val_overall_acc, torch.Tensor):
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

