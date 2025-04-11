import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
import shutil
import json
import argparse

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="ModelNet40_12View/*/train")
parser.add_argument("-val_path", type=str, default="ModelNet40_12View/*/test")
parser.add_argument("-epoch", '-e', type=int, help="number of epochs", default=30)
parser.set_defaults(train=False)

def create_folder(log_dir):
    if os.path.exists(log_dir):
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()

    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    with open(os.path.join(log_dir, 'config.json'), 'w') as config_f:
        json.dump(vars(args), config_f)
        
    print('*'*50, "Settings:", '*'*50)
    print(f"Name: {args.name}, Num Model: {args.num_models}")
    print(f"Learning Rate: {args.lr}, Weight Decay: {args.weight_decay}, Pretraining: {pretraining}")
    print(f"CNN Name: {args.cnn_name}, Num Views: {args.num_views}")
    print(f"Train Path: {args.train_path}, Val Path: {args.val_path}, Epochs: {args.epochs}")
    
    # STAGE 1
    log_dir = f'{args.name}_stage_1'
    create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)

    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_models_train = args.num_models*args.num_views

    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f'num_train_files: {len(train_dataset.filepaths)}')
    print(f'num_val_files: {len(val_dataset.filepaths)}')
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
    trainer.train(args.epoch)
    torch.cuda.empty_cache()  # Clear GPU memory before stage 2
    # Save the final model
    cnet.save(log_dir, args.epoch)

    # STAGE 2
    log_dir = f'{args.name}_stage_2'
    create_folder(log_dir)
    cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    del cnet

    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)# shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    print(f'num_train_files: {len(train_dataset.filepaths)}')
    print(f'num_val_files: {len(val_dataset.filepaths)}')
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
    trainer.train(args.epoch)
    torch.cuda.empty_cache()  # Clear GPU memory after training
    # Save the final model
    cnet_2.save(log_dir, args.epoch)